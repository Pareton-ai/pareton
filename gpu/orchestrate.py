"""Provision + bootstrap + remote bench + teardown."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import timedelta, timezone
from pathlib import Path

import config
from bench.correctness import resolve_trace_path
from bench.validate import (
    RequestValidationError,
    load_bench_request,
    load_workload_trace,
)

from gpu.bootstrap import (
    REMOTE_HF_CACHE,
    REMOTE_REPO,
    REMOTE_VENV,
    bootstrap_pod,
    pull_engine_images,
)
from gpu.errors import DestroyError, GpuError, ProvisionError
from gpu.keys import ensure_durable_keypair, read_public_key
from gpu.providers import get_provider
from gpu.registry import (
    PodRegistry,
    RegistryEntry,
    encode_pod_name,
    parse_pod_name,
)
from gpu.ssh import REPO_RSYNC_EXCLUDES, SshRunner, exec as ssh_exec, pull, push
from gpu.types import Pod, PodSpec

logger = logging.getLogger(__name__)

# Under /opt/pareton so the (often non-root) Targon SSH user can write/source it.
REMOTE_ENV = f"{REMOTE_REPO}/.pareton-bench.env"
REMOTE_OUT = f"{REMOTE_REPO}/out"
REMOTE_REQUEST = f"{REMOTE_REPO}/bench_request.remote.json"
REMOTE_TRACE_DIR = f"{REMOTE_REPO}/.pareton-traces"

# Teardown failed (pod/volume may still be billing). Distinct from CLI/preflight 2
# and from typical bench failure codes so CI can alert without log scraping.
EXIT_DESTROY_FAILED = 75


def _path_excluded_from_repo_rsync(rel: Path) -> bool:
    """True if rel (under repo root) would not be shipped by bootstrap rsync."""
    parts = rel.parts
    name = parts[-1] if parts else ""
    for pattern in REPO_RSYNC_EXCLUDES:
        if pattern.endswith("/"):
            top = pattern.rstrip("/")
            if parts and parts[0] == top:
                return True
        elif pattern.startswith("*") and name and Path(name).match(pattern):
            return True
        elif pattern in parts or (parts and parts[0] == pattern):
            return True
    return False


def _trace_needs_explicit_push(trace_path: Path, repo_root: Path) -> bool:
    """Push when outside the repo or under an rsync-excluded prefix (e.g. out/)."""
    try:
        rel = trace_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return True
    return _path_excluded_from_repo_rsync(rel)


def _repo_root() -> Path:
    try:
        import config as _cfg

        return Path(getattr(_cfg, "REPO_ROOT", Path(__file__).resolve().parents[1]))
    except Exception:  # noqa: BLE001
        return Path(__file__).resolve().parents[1]


def _select_offer(provider, spec: PodSpec):
    offers = provider.search(spec)
    if not offers:
        raise ProvisionError(
            f"no offers from {provider.name} matching "
            f"gpu_type={spec.gpu_type!r} count>={spec.gpu_count} "
            f"max_hourly_cents={spec.max_hourly_cents}"
        )
    return offers[0]


def _write_remote_env(
    pod: Pod, *, runner: SshRunner | None, state_dir: Path | None
) -> None:
    lines: list[str] = [f"PARETON_BENCH_HF_CACHE_DIR={REMOTE_HF_CACHE}"]
    hf = os.environ.get("HF_TOKEN") or os.environ.get("PARETON_HF_TOKEN")
    if hf:
        lines.append(f"HF_TOKEN={hf}")
    ghcr_token = os.environ.get("PARETON_GHCR_TOKEN", "")
    ghcr_user = (
        os.environ.get("PARETON_GHCR_USER")
        or os.environ.get("PARETON_GHCR_USERNAME")
        or ""
    )
    if ghcr_token:
        lines.append(f"PARETON_GHCR_TOKEN={ghcr_token}")
        if ghcr_user:
            lines.append(f"PARETON_GHCR_USER={ghcr_user}")
    payload = "\n".join(lines) + "\n"
    # Push via rsync so tokens never appear in ssh remote argv / local logs.
    fd, tmp_name = tempfile.mkstemp(prefix="pareton-bench-env-")
    local = Path(tmp_name)
    try:
        os.write(fd, payload.encode("utf-8"))
        os.close(fd)
        fd = -1
        os.chmod(local, 0o600)
        push(
            pod,
            local,
            REMOTE_ENV,
            excludes=[],
            runner=runner,
            state_dir=state_dir,
        )
        ssh_exec(
            pod,
            f"chmod 600 {REMOTE_ENV}",
            timeout_s=30.0,
            runner=runner,
            state_dir=state_dir,
        )
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            local.unlink(missing_ok=True)
        except OSError:
            pass


def _delete_remote_env(
    pod: Pod, *, runner: SshRunner | None, state_dir: Path | None
) -> None:
    try:
        ssh_exec(
            pod,
            f"rm -f {REMOTE_ENV}",
            timeout_s=30.0,
            runner=runner,
            state_dir=state_dir,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to delete remote env file: %s", exc)


def _entry_from_pod(pod: Pod, *, state: str = "active") -> RegistryEntry:
    parsed = parse_pod_name(pod.name)
    if parsed:
        created, ttl, deadline = parsed
    else:
        created = pod.created_utc
        ttl = pod.ttl_hours
        deadline = created + timedelta(hours=ttl or 2.0)
    return RegistryEntry(
        provider=pod.provider,
        pod_id=pod.pod_id,
        name=pod.name,
        deadline=deadline.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        hourly_price_cents=pod.hourly_price_cents,
        volume_uid=str((pod.raw or {}).get("volume_uid", "")),
        volume_name=str((pod.raw or {}).get("volume_name", "")),
        state=state,
        key_path=str(pod.key_path),
        ssh_host=pod.ssh.host,
        ssh_port=pod.ssh.port,
        ssh_user=pod.ssh.user,
        created_utc=created.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ttl_hours=float(ttl),
        raw=pod.raw,
    )


def _engine_image_refs(req) -> list[str]:
    refs: list[str] = []
    for eng in (req.engines.baseline, req.engines.candidate):
        img = str(eng.image or "").strip()
        if img and img not in refs:
            refs.append(img)
    return refs


def provision_pod(
    spec: PodSpec,
    *,
    registry: PodRegistry | None = None,
    provider=None,
    state_dir: Path | None = None,
) -> Pod:
    registry = registry or PodRegistry(state_dir)
    if provider is None:
        provider = get_provider(spec.provider, state_dir=registry.state_dir)

    ensure_durable_keypair(registry.state_dir)
    pub = read_public_key(registry.state_dir)

    def _do_provision() -> Pod:
        if provider.name != "static_ssh" and not spec.force:
            blocking = registry.has_blocking_managed()
            if blocking is not None:
                raise ProvisionError(
                    f"single-flight: registry already has {blocking.state} pod "
                    f"{blocking.name} ({blocking.provider}); pass --force to override"
                )
        offer = _select_offer(provider, spec)
        pod_name = encode_pod_name(ttl_hours=spec.ttl_hours)
        if spec.manual:
            if provider.name == "static_ssh":
                raise ProvisionError("--manual is not supported for static_ssh")
            provision_fn = getattr(provider, "provision_manual", None)
            if provision_fn is None:
                raise ProvisionError(
                    f"provider {provider.name!r} does not support --manual"
                )
            pod = provision_fn(offer, name=pod_name, ssh_public_key=pub)
        else:
            pod = provider.provision(offer, name=pod_name, ssh_public_key=pub)
        pod.ttl_hours = spec.ttl_hours
        if provider.name == "static_ssh":
            return pod
        try:
            registry.add(_entry_from_pod(pod, state="active"))
        except Exception as exc:
            # Cloud rent succeeded; must not leave a billable orphan without a handle.
            try:
                provider.destroy(pod)
            except Exception as destroy_exc:  # noqa: BLE001
                logger.error(
                    "registry.add failed after rent AND destroy failed for "
                    "pod=%s volume=%s: add=%s destroy=%s. Destroy manually NOW.",
                    pod.name,
                    (pod.raw or {}).get("volume_uid", ""),
                    exc,
                    destroy_exc,
                )
                raise ProvisionError(
                    f"registry.add failed after rent ({exc}); destroy also failed "
                    f"({destroy_exc}); manual cleanup required for {pod.name} "
                    f"volume={(pod.raw or {}).get('volume_uid', '')}"
                ) from destroy_exc
            raise ProvisionError(
                f"registry.add failed after rent; cloud resource destroyed: {exc}"
            ) from exc
        return pod

    if provider.name == "static_ssh":
        return _do_provision()
    with registry.provision_lock():
        return _do_provision()


def destroy_pod(
    pod: Pod,
    *,
    registry: PodRegistry | None = None,
    provider=None,
    state_dir: Path | None = None,
) -> None:
    registry = registry or PodRegistry(state_dir)
    if provider is None:
        provider = get_provider(pod.provider, state_dir=registry.state_dir)
    try:
        provider.destroy(pod)
    except DestroyError:
        if pod.provider != "static_ssh":
            try:
                registry.update(_entry_from_pod(pod, state="destroy_failed"))
            except Exception as reg_exc:  # noqa: BLE001
                logger.error(
                    "registry update failed after destroy failure for %s: %s",
                    pod.name,
                    reg_exc,
                )
        raise
    if pod.provider != "static_ssh":
        registry.remove(pod.name)


def run_bench_on_pod(
    spec: PodSpec,
    *,
    request_path: Path,
    output_dir: Path,
    mock_engine: bool = False,
    repetitions: int = 1,
    registry: PodRegistry | None = None,
    provider=None,
    runner: SshRunner | None = None,
    state_dir: Path | None = None,
    repo_root: Path | None = None,
) -> int:
    """Full provision -> bench -> tear down.

    When repetitions > 1, runs the remote harness sequentially into distinct
    run-001..run-NNN directories after a single provision/bootstrap/image pull.

    Returns the bench exit code, or EXIT_DESTROY_FAILED (75) if teardown failed
    (pod/volume may still be billing; takes precedence over the bench code).
    """
    if repetitions < 1:
        raise ProvisionError(f"repetitions must be >= 1, got {repetitions}")

    registry = registry or PodRegistry(state_dir)
    repo_root = (repo_root or _repo_root()).resolve()
    request_path = Path(request_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preflight BEFORE rent (shape + existence + digest).
    try:
        req, _raw = load_bench_request(request_path)
        trace_path = resolve_trace_path(
            req.workload_trace.path, request_path=request_path
        )
        load_workload_trace(trace_path, expected_sha256=req.workload_trace.sha256)
    except RequestValidationError as exc:
        raise ProvisionError(f"bench request preflight failed: {exc}") from exc

    pod: Pod | None = None
    exit_code = 1
    destroy_failed = False
    pending: BaseException | None = None
    if provider is None:
        provider = get_provider(spec.provider, state_dir=registry.state_dir)

    try:
        pod = provision_pod(
            spec, registry=registry, provider=provider, state_dir=registry.state_dir
        )
        code_sha = bootstrap_pod(
            pod,
            repo_root=repo_root,
            runner=runner,
            state_dir=registry.state_dir,
        )

        # Prepare remote request (+ push trace when bootstrap rsync would omit it).
        remote_req = json.loads(request_path.read_text(encoding="utf-8"))
        if _trace_needs_explicit_push(trace_path, repo_root):
            ssh_exec(
                pod,
                f"mkdir -p {REMOTE_TRACE_DIR}",
                timeout_s=60.0,
                runner=runner,
                state_dir=registry.state_dir,
            )
            remote_trace = f"{REMOTE_TRACE_DIR}/{trace_path.name}"
            push(
                pod,
                trace_path,
                remote_trace,
                excludes=[],
                runner=runner,
                state_dir=registry.state_dir,
            )
            remote_req["workload_trace"]["path"] = remote_trace
        else:
            rel = trace_path.resolve().relative_to(repo_root.resolve()).as_posix()
            remote_req["workload_trace"]["path"] = f"{REMOTE_REPO}/{rel}"

        local_remote_req = output_dir / "bench_request.remote.json"
        local_remote_req.write_text(
            json.dumps(remote_req, indent=2) + "\n", encoding="utf-8"
        )
        push(
            pod,
            local_remote_req,
            REMOTE_REQUEST,
            excludes=[],
            runner=runner,
            state_dir=registry.state_dir,
        )

        _write_remote_env(pod, runner=runner, state_dir=registry.state_dir)

        if not mock_engine:
            pull_engine_images(
                pod,
                _engine_image_refs(req),
                env_file=REMOTE_ENV,
                runner=runner,
                state_dir=registry.state_dir,
            )

        mock_flag = " --mock-engine" if mock_engine else ""
        for i in range(1, repetitions + 1):
            if repetitions == 1:
                remote_out = REMOTE_OUT
                local_out = output_dir
            else:
                run_name = f"run-{i:03d}"
                remote_out = f"{REMOTE_OUT}/{run_name}"
                local_out = output_dir / run_name
                local_out.mkdir(parents=True, exist_ok=True)

            bench_cmd = (
                f"cd {REMOTE_REPO} && set -a && . {REMOTE_ENV} && set +a && "
                f"export PARETON_BENCH_CODE_SHA={code_sha} && "
                f"mkdir -p {remote_out} && "
                f"{REMOTE_VENV}/bin/python -m bench "
                f"--request {REMOTE_REQUEST} --output-dir {remote_out}{mock_flag}"
            )
            result = ssh_exec(
                pod,
                bench_cmd,
                timeout_s=float(config.BENCH_TIMEOUT_S),
                runner=runner,
                state_dir=registry.state_dir,
                check=False,
            )
            if result.stdout:
                print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
            if result.stderr:
                print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")

            try:
                pull(
                    pod,
                    f"{remote_out}/",
                    local_out,
                    runner=runner,
                    state_dir=registry.state_dir,
                )
            except GpuError as exc:
                logger.warning("failed to pull bench output: %s", exc)

            exit_code = int(result.exit_code)
            if exit_code != 0:
                break
    except BaseException as exc:
        pending = exc
    finally:
        if pod is not None:
            _delete_remote_env(pod, runner=runner, state_dir=registry.state_dir)
            try:
                destroy_pod(
                    pod,
                    registry=registry,
                    provider=provider,
                    state_dir=registry.state_dir,
                )
            except DestroyError as exc:
                destroy_failed = True
                logger.error(
                    "DESTROY FAILED for pod=%s volume=%s provider=%s: %s. "
                    "Destroy manually NOW in the provider dashboard "
                    "(https://targon.com/rentals).",
                    pod.name,
                    (pod.raw or {}).get("volume_uid", ""),
                    pod.provider,
                    exc,
                )
                print(
                    f"ERROR: destroy failed for {pod.name} "
                    f"volume={(pod.raw or {}).get('volume_uid', '')} - "
                    f"destroy manually NOW on the {pod.provider} dashboard",
                    flush=True,
                )

    if destroy_failed:
        return EXIT_DESTROY_FAILED
    if pending is not None:
        raise pending
    return exit_code
