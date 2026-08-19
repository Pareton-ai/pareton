"""Provision + bootstrap + remote bench + teardown."""

from __future__ import annotations

import json
import logging
import os
import shlex
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Sequence

import config
from bench.correctness import resolve_trace_path
from bench.output import PHASE_FILENAME
from bench.phases import POD_REPORTABLE_PHASES, BenchPhase, coerce_phase
from bench.validate import (
    RequestValidationError,
    load_bench_request,
    load_workload_trace,
)

from gpu.bootstrap import (
    REMOTE_ENGINE_CACHE,
    REMOTE_HF_CACHE,
    REMOTE_REPO,
    REMOTE_VENV,
    bootstrap_pod,
    pull_engine_images,
)
from gpu.errors import DestroyError, GpuError, ProvisionError


from gpu.keys import ensure_durable_keypair, read_public_key
from gpu.providers import get_provider, provider_order
from gpu.registry import (
    PodRegistry,
    RegistryEntry,
    encode_pod_name,
    parse_pod_name,
)
from gpu.ssh import REPO_RSYNC_EXCLUDES, SshRunner, exec as ssh_exec, pull, push
from gpu.types import Pod, PodSpec, SshTarget
from observability import events as obs

logger = logging.getLogger(__name__)


class RegistryAddError(GpuError):
    """registry.add failed after a successful rent; not a provider failure."""


PhaseSink = Callable[[str], None]


def _noop_phase(_phase: str) -> None:
    return None


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
    lines: list[str] = [
        f"PARETON_BENCH_HF_CACHE_DIR={REMOTE_HF_CACHE}",
        f"PARETON_BENCH_ENGINE_CACHE_DIR={REMOTE_ENGINE_CACHE}",
        # Harness reads config on the pod, which has no .env; forward the
        # worker-side value so large models get the same health window.
        f"PARETON_BENCH_HEALTH_TIMEOUT_S={config.BENCH_HEALTH_TIMEOUT_S}",
        # GPU pods have the RAM/bandwidth for Xet HP mode (>=64 GB). Hub >=0.32
        # already uses hf_xet; this only raises concurrency/buffers. Do not set
        # on laptops (can be slower with less RAM).
        "HF_XET_HIGH_PERFORMANCE=1",
    ]
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
    """Rent one pod, falling back through the configured provider order.

    With no injected ``provider``, the primary comes from ``spec.provider``
    and fallbacks from ``PARETON_GPU_PROVIDER_FALLBACKS`` (default shadeform).
    A ``ProvisionError`` (no capacity, rent/wait-ready failure) tries the next
    provider; any other exception aborts immediately. The single-flight check
    is local policy and never triggers fallback.
    """
    registry = registry or PodRegistry(state_dir)
    if provider is not None:
        providers = [provider]
    else:
        providers = []
        order = provider_order(spec.provider)
        for name in order:
            try:
                providers.append(get_provider(name, state_dir=registry.state_dir))
            except ProvisionError as exc:
                logger.warning("provider %s unavailable, skipping: %s", name, exc)
                obs.pod_provision_failed(provider=name, error=str(exc))
        if not providers:
            raise ProvisionError(f"no usable GPU provider (order: {', '.join(order)})")

    ensure_durable_keypair(registry.state_dir)
    pub = read_public_key(registry.state_dir)

    def _do_provision(p) -> Pod:
        offer = _select_offer(p, spec)
        pod_name = encode_pod_name(ttl_hours=spec.ttl_hours)
        if spec.manual:
            if p.name == "static_ssh":
                raise ProvisionError("--manual is not supported for static_ssh")
            provision_fn = getattr(p, "provision_manual", None)
            if provision_fn is None:
                raise ProvisionError(f"provider {p.name!r} does not support --manual")
            pod = provision_fn(offer, name=pod_name, ssh_public_key=pub)
        else:
            pod = p.provision(offer, name=pod_name, ssh_public_key=pub)
        pod.ttl_hours = spec.ttl_hours
        if p.name == "static_ssh":
            return pod
        try:
            registry.add(_entry_from_pod(pod, state="active"))
        except Exception as exc:
            # Cloud rent succeeded; must not leave a billable orphan without a handle.
            try:
                p.destroy(pod)
            except Exception as destroy_exc:  # noqa: BLE001
                logger.error(
                    "registry.add failed after rent AND destroy failed for "
                    "pod=%s volume=%s: add=%s destroy=%s. Destroy manually NOW.",
                    pod.name,
                    (pod.raw or {}).get("volume_uid", ""),
                    exc,
                    destroy_exc,
                )
                raise RegistryAddError(
                    f"registry.add failed after rent ({exc}); destroy also failed "
                    f"({destroy_exc}); manual cleanup required for {pod.name} "
                    f"volume={(pod.raw or {}).get('volume_uid', '')}"
                ) from destroy_exc
            raise RegistryAddError(
                f"registry.add failed after rent; cloud resource destroyed: {exc}"
            ) from exc
        return pod

    def _attempt_providers() -> Pod:
        if providers[0].name != "static_ssh" and not spec.force:
            blocking = registry.has_blocking_managed()
            if blocking is not None:
                raise ProvisionError(
                    f"single-flight: registry already has {blocking.state} pod "
                    f"{blocking.name} ({blocking.provider}); pass --force to override"
                )
        last_exc: ProvisionError | None = None
        for i, p in enumerate(providers):
            try:
                return _do_provision(p)
            except RegistryAddError:
                # Local registry failure after rent: never fall back and rent
                # a second pod while the first may still be untracked.
                raise
            except ProvisionError as exc:
                last_exc = exc
                obs.pod_provision_failed(provider=p.name, error=str(exc))
                if i + 1 < len(providers):
                    logger.warning(
                        "provision failed on %s (%s); falling back to %s",
                        p.name,
                        exc,
                        providers[i + 1].name,
                    )
            except Exception as exc:
                obs.pod_provision_failed(provider=p.name, error=str(exc))
                raise
        assert last_exc is not None
        raise last_exc

    if providers[0].name == "static_ssh":
        result_pod = _attempt_providers()
    else:
        with registry.provision_lock():
            result_pod = _attempt_providers()
    obs.pod_provisioned(pod=result_pod.name, provider=result_pod.provider)
    return result_pod


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
    except DestroyError as exc:
        obs.destroy_failed(
            pod=pod.name,
            provider=pod.provider,
            error=str(exc),
            volume_uid=str((pod.raw or {}).get("volume_uid", "")),
        )
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
    obs.pod_destroyed(pod=pod.name, provider=pod.provider)
    if pod.provider != "static_ssh":
        registry.remove(pod.name)


def _report_is_pass(run_dir: Path) -> bool:
    report = Path(run_dir) / "bench_report.json"
    if not report.is_file():
        return False
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return str(data.get("verdict", "")).lower() == "pass"


def _pod_from_entry(entry: RegistryEntry) -> Pod:
    return Pod(
        provider=entry.provider,
        pod_id=entry.pod_id,
        name=entry.name,
        ssh=SshTarget(
            host=entry.ssh_host,
            port=entry.ssh_port,
            user=entry.ssh_user,
        ),
        key_path=Path(entry.key_path),
        hourly_price_cents=entry.hourly_price_cents,
        created_utc=datetime.now(timezone.utc),
        ttl_hours=entry.ttl_hours,
        raw={
            "volume_uid": entry.volume_uid,
            "volume_name": entry.volume_name,
        },
    )


def _bench_jobs(
    *,
    request_path: Path | None,
    request_paths: Sequence[Path] | None,
    output_dir: Path,
    repetitions: int,
) -> list[tuple[Path, Path]]:
    if request_paths is not None:
        if request_path is not None:
            raise ProvisionError("provide --request or --requests-dir, not both")
        if repetitions != 1:
            raise ProvisionError("--requests-dir is incompatible with --repetitions")
        return [
            (Path(p).resolve(), output_dir / f"run-{i:03d}")
            for i, p in enumerate(request_paths, start=1)
        ]
    if request_path is None:
        raise ProvisionError("provide --request or --requests-dir")
    resolved = Path(request_path).resolve()
    if repetitions == 1:
        return [(resolved, output_dir)]
    return [(resolved, output_dir / f"run-{i:03d}") for i in range(1, repetitions + 1)]


def _preflight_request(request_path: Path, repo_root: Path):
    req, _raw = load_bench_request(request_path)
    trace_path = resolve_trace_path(req.workload_trace.path, request_path=request_path)
    load_workload_trace(trace_path, expected_sha256=req.workload_trace.sha256)
    return req, trace_path


def _push_remote_request(
    pod: Pod,
    *,
    request_path: Path,
    trace_path: Path,
    local_out: Path,
    repo_root: Path,
    runner: SshRunner | None,
    state_dir: Path | None,
) -> None:
    remote_req = json.loads(request_path.read_text(encoding="utf-8"))
    if _trace_needs_explicit_push(trace_path, repo_root):
        ssh_exec(
            pod,
            f"mkdir -p {REMOTE_TRACE_DIR}",
            timeout_s=60.0,
            runner=runner,
            state_dir=state_dir,
        )
        remote_trace = f"{REMOTE_TRACE_DIR}/{trace_path.name}"
        push(
            pod,
            trace_path,
            remote_trace,
            excludes=[],
            runner=runner,
            state_dir=state_dir,
        )
        remote_req["workload_trace"]["path"] = remote_trace
    else:
        rel = trace_path.resolve().relative_to(repo_root.resolve()).as_posix()
        remote_req["workload_trace"]["path"] = f"{REMOTE_REPO}/{rel}"

    local_out.mkdir(parents=True, exist_ok=True)
    local_remote_req = local_out / "bench_request.remote.json"
    local_remote_req.write_text(
        json.dumps(remote_req, indent=2) + "\n", encoding="utf-8"
    )
    push(
        pod,
        local_remote_req,
        REMOTE_REQUEST,
        excludes=[],
        runner=runner,
        state_dir=state_dir,
    )


def read_pod_phase(
    pod: Pod,
    *,
    remote_out: str,
    runner: SshRunner | None = None,
    state_dir: Path | None = None,
    max_bytes: int = 4096,
) -> str | None:
    """Harness phase on the pod, or None. Untrusted: byte-capped and vocabulary-checked."""
    path = shlex.quote(f"{remote_out}/{PHASE_FILENAME}")
    result = ssh_exec(
        pod,
        f"head -c {int(max_bytes)} {path} 2>/dev/null || true",
        timeout_s=30.0,
        runner=runner,
        state_dir=state_dir,
        check=False,
    )
    if result.exit_code != 0:
        return None
    try:
        record = json.loads((result.stdout or "").strip() or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None
    return coerce_phase(record.get("phase"), allowed=POD_REPORTABLE_PHASES)


class _PodPhasePoller:
    """Poll remote phase.json while ssh exec blocks. Failures leave the last phase in place."""

    def __init__(
        self,
        pod: Pod,
        *,
        remote_out: str,
        on_phase: PhaseSink,
        runner: SshRunner | None = None,
        state_dir: Path | None = None,
        interval_s: float | None = None,
    ) -> None:
        self._pod = pod
        self._remote_out = remote_out
        self._on_phase = on_phase
        self._runner = runner
        self._state_dir = state_dir
        self._interval_s = (
            config.BENCH_PHASE_POLL_S if interval_s is None else interval_s
        )
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def poll_once(self) -> None:
        try:
            phase = read_pod_phase(
                self._pod,
                remote_out=self._remote_out,
                runner=self._runner,
                state_dir=self._state_dir,
            )
        except Exception as exc:  # noqa: BLE001 - progress must not fail a bench
            logger.debug("pod phase poll failed: %s", exc)
            return
        if phase is None:
            return
        with self._lock:
            # Join can time out while SSH is still in flight; never overwrite
            # a later worker-owned phase such as teardown.
            if self._stop.is_set():
                return
            self._on_phase(phase)

    def _loop(self) -> None:
        self.poll_once()
        while not self._stop.wait(self._interval_s):
            self.poll_once()

    def __enter__(self) -> _PodPhasePoller:
        self._thread = threading.Thread(
            target=self._loop, name="pod-phase-poll", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        with self._lock:
            self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5.0)


def run_bench_on_pod(
    spec: PodSpec,
    *,
    request_path: Path | None = None,
    request_paths: Sequence[Path] | None = None,
    output_dir: Path,
    mock_engine: bool = False,
    repetitions: int = 1,
    pod_name: str | None = None,
    keep: bool = False,
    registry: PodRegistry | None = None,
    provider=None,
    runner: SshRunner | None = None,
    state_dir: Path | None = None,
    repo_root: Path | None = None,
    on_phase: PhaseSink | None = None,
) -> int:
    """Run bench request(s) on a GPU pod.

    Default: provision -> bench -> destroy. ``--pod`` reuses a registry pod.
    ``--keep`` skips destroy. ``request_paths`` (from ``--requests-dir``) runs
    each sample-N request into output_dir/run-00N and skips local pass reports.

    ``--repetitions`` still replays one request N times (not a sample pool).

    ``on_phase`` is called as the run moves through provisioning, bootstrap, pull, harness, teardown.

    Returns the last bench exit code, or EXIT_DESTROY_FAILED (75) if teardown
    failed (pod/volume may still be billing; takes precedence).
    """
    if repetitions < 1:
        raise ProvisionError(f"repetitions must be >= 1, got {repetitions}")
    phase = on_phase or _noop_phase

    registry = registry or PodRegistry(state_dir)
    repo_root = (repo_root or _repo_root()).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = request_paths is not None
    jobs = _bench_jobs(
        request_path=request_path,
        request_paths=request_paths,
        output_dir=output_dir,
        repetitions=repetitions,
    )
    pending_jobs: list[tuple[Path, Path]] = []
    for req_p, local_out in jobs:
        if pool and _report_is_pass(local_out):
            logger.info("skip %s (already pass)", local_out)
            continue
        pending_jobs.append((req_p, local_out))
    if not pending_jobs:
        logger.info("all %d sample(s) already pass; nothing to run", len(jobs))
        return 0

    preflighted: list[tuple[Path, Path, object, Path]] = []
    try:
        for req_p, local_out in pending_jobs:
            req, trace_path = _preflight_request(req_p, repo_root)
            preflighted.append((req_p, local_out, req, trace_path))
    except RequestValidationError as exc:
        raise ProvisionError(f"bench request preflight failed: {exc}") from exc

    pod: Pod | None = None
    exit_code = 1
    destroy_failed = False
    pending: BaseException | None = None

    try:
        if pod_name:
            entry = registry.get(pod_name)
            if entry is None:
                raise ProvisionError(f"unknown pod {pod_name!r} in registry")
            pod = _pod_from_entry(entry)
            logger.info("reusing pod %s", pod.name)
        else:
            # provider=None: provision_pod walks the configured fallback order
            # and destroy_pod later resolves the real provider from pod.provider.
            phase(BenchPhase.PROVISIONING.value)
            pod = provision_pod(
                spec,
                registry=registry,
                provider=provider,
                state_dir=registry.state_dir,
            )
        phase(BenchPhase.BOOTSTRAPPING.value)
        code_sha = bootstrap_pod(
            pod,
            repo_root=repo_root,
            runner=runner,
            state_dir=registry.state_dir,
        )
        _write_remote_env(pod, runner=runner, state_dir=registry.state_dir)

        if not mock_engine:
            refs: list[str] = []
            for _req_p, _local_out, req, _trace in preflighted:
                for ref in _engine_image_refs(req):
                    if ref not in refs:
                        refs.append(ref)
            phase(BenchPhase.PULLING_IMAGE.value)
            pull_engine_images(
                pod,
                refs,
                env_file=REMOTE_ENV,
                runner=runner,
                state_dir=registry.state_dir,
            )

        mock_flag = " --mock-engine" if mock_engine else ""
        for req_p, local_out, _req, trace_path in preflighted:
            local_out.mkdir(parents=True, exist_ok=True)
            if pool or repetitions > 1:
                run_name = local_out.name
                remote_out = f"{REMOTE_OUT}/{run_name}"
            else:
                remote_out = REMOTE_OUT

            _push_remote_request(
                pod,
                request_path=req_p,
                trace_path=trace_path,
                local_out=local_out,
                repo_root=repo_root,
                runner=runner,
                state_dir=registry.state_dir,
            )

            bench_cmd = (
                f"cd {REMOTE_REPO} && set -a && . {REMOTE_ENV} && set +a && "
                f"export PARETON_BENCH_CODE_SHA={code_sha} && "
                f"mkdir -p {remote_out} && "
                f"{REMOTE_VENV}/bin/python -m bench "
                f"--request {REMOTE_REQUEST} --output-dir {remote_out}{mock_flag}"
            )
            # ssh exec does not stream; poll the harness marker while it blocks.
            with _PodPhasePoller(
                pod,
                remote_out=remote_out,
                on_phase=phase,
                runner=runner,
                state_dir=registry.state_dir,
            ):
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
            if keep:
                print(f"keep pod={pod.name}", flush=True)
            else:
                phase(BenchPhase.TEARDOWN.value)
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
