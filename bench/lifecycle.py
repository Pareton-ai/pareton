"""Docker engine lifecycle: network, start, health-check, teardown.

Engine-as-black-box: given an image (by digest or local tag), yield a healthy
OpenAI-compatible base URL on an isolated Docker network, then always tear down
and capture container logs.

Talks to Docker via the CLI (subprocess), matching builder/hermetic.py. Unit
tests inject a fake runner — no Docker required for the default suite.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.error import URLError
from urllib.request import urlopen

from bench.schemas import EngineSpec

logger = logging.getLogger(__name__)

NAME_PREFIX = "pareton-bench-"

# Default timeouts (overridden by config when available; kept local so bench/
# stays importable on a fresh pod without depending on config for core paths).
_DEFAULT_HEALTH_TIMEOUT_S = 600.0
_DEFAULT_HEALTH_POLL_S = 2.0
_DEFAULT_ENGINE_PORT = 8000
_DEFAULT_PULL_TIMEOUT_S = 1800.0
_DEFAULT_CMD_TIMEOUT_S = 120.0


def ensure_listen_args(serve_args: Sequence[str], port: int) -> list[str]:
    """Pin listen address so health checks on the docker network IP succeed.

    SGLang defaults to 127.0.0.1:30000. The harness probes ``{container_ip}:{port}``
    (default 8000). Without these flags the server comes up and the health loop
    still sees connection refused until timeout.
    """
    args = list(serve_args)
    if "--host" not in args:
        args.extend(["--host", "0.0.0.0"])
    if "--port" not in args:
        args.extend(["--port", str(int(port))])
    return args


def _defaults_from_config() -> tuple[float, float, int, float, float]:
    try:
        import config as _cfg

        return (
            float(getattr(_cfg, "BENCH_HEALTH_TIMEOUT_S", _DEFAULT_HEALTH_TIMEOUT_S)),
            float(getattr(_cfg, "BENCH_HEALTH_POLL_S", _DEFAULT_HEALTH_POLL_S)),
            int(getattr(_cfg, "BENCH_ENGINE_PORT", _DEFAULT_ENGINE_PORT)),
            float(
                getattr(_cfg, "BENCH_DOCKER_PULL_TIMEOUT_S", _DEFAULT_PULL_TIMEOUT_S)
            ),
            float(getattr(_cfg, "BENCH_DOCKER_CMD_TIMEOUT_S", _DEFAULT_CMD_TIMEOUT_S)),
        )
    except Exception:  # noqa: BLE001 — standalone pod may lack config
        return (
            _DEFAULT_HEALTH_TIMEOUT_S,
            _DEFAULT_HEALTH_POLL_S,
            _DEFAULT_ENGINE_PORT,
            _DEFAULT_PULL_TIMEOUT_S,
            _DEFAULT_CMD_TIMEOUT_S,
        )


class EngineError(Exception):
    """Engine lifecycle failure (maps to CLI exit code 3 in B4 wiring)."""

    def __init__(self, message: str = "", *, error_role: str | None = None) -> None:
        super().__init__(message)
        self.error_role = error_role


class HostEnvironmentError(Exception):
    """Host/tooling unavailable (maps to CLI exit code 2): Docker missing/down."""


_DAEMON_UNAVAILABLE_SNIPPETS = (
    "cannot connect to the docker daemon",
    "is the docker daemon running",
    "failed to connect to the docker api",
    "permission denied while trying to connect to the docker daemon",
)


def raise_docker_failure(prefix: str, detail: str) -> None:
    """Raise HostEnvironmentError for daemon-down; otherwise EngineError."""
    text = (detail or "").strip()
    lower = text.lower()
    if any(s in lower for s in _DAEMON_UNAVAILABLE_SNIPPETS):
        raise HostEnvironmentError(f"{prefix}: {text}")
    raise EngineError(f"{prefix}: {text}")


@dataclass
class DockerResult:
    returncode: int
    stdout: str
    stderr: str


DockerRunner = Callable[..., DockerResult]


def _redact_cmd_for_log(cmd: Sequence[str]) -> list[str]:
    """Redact values after -e/--env and contents of --env-file paths in logs."""
    out: list[str] = []
    i = 0
    while i < len(cmd):
        arg = cmd[i]
        if arg in ("-e", "--env") and i + 1 < len(cmd):
            key_val = cmd[i + 1]
            key = key_val.split("=", 1)[0] if "=" in key_val else key_val
            out.extend([arg, f"{key}=***"])
            i += 2
            continue
        if arg == "--env-file" and i + 1 < len(cmd):
            out.extend([arg, "<redacted>"])
            i += 2
            continue
        out.append(arg)
        i += 1
    return out


def default_docker_runner(
    cmd: Sequence[str],
    *,
    timeout: float,
    input_text: str | None = None,
) -> DockerResult:
    """Run a docker CLI command with a hard timeout. Never logs env values."""
    if not cmd or cmd[0] != "docker":
        raise EngineError(f"docker runner expected docker argv, got {cmd!r}")
    logger.info("docker: %s", " ".join(_redact_cmd_for_log(cmd)))
    try:
        proc = subprocess.run(
            list(cmd),
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise EngineError(
            f"docker command timed out after {timeout}s: "
            f"{' '.join(_redact_cmd_for_log(cmd))}"
        ) from exc
    except FileNotFoundError as exc:
        raise HostEnvironmentError("docker CLI not found on PATH") from exc
    return DockerResult(
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


def new_run_id() -> str:
    """Short uuid suffix for pareton-bench-<run_id> names."""
    return uuid.uuid4().hex[:12]


def extract_digest_from_image_ref(image: str) -> str | None:
    """If image is pinned as name@sha256:..., return sha256:... else None."""
    if "@sha256:" in image:
        digest = image.rsplit("@", 1)[1]
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            return digest
    return None


def normalize_image_id(image_id: str) -> str:
    """Normalize docker image Id to sha256:<64hex> form."""
    image_id = image_id.strip()
    if image_id.startswith("sha256:"):
        return image_id
    if re.fullmatch(r"[0-9a-f]{64}", image_id):
        return f"sha256:{image_id}"
    # docker may return sha256: truncated or with prefix noise
    if "sha256:" in image_id:
        return "sha256:" + image_id.split("sha256:", 1)[1][:64]
    return image_id


@dataclass
class EngineHandle:
    base_url: str
    container_id: str
    container_name: str
    image_digest: str
    image: str


@dataclass
class BenchNetwork:
    """Docker network create [--internal] pareton-bench-<run_id>."""

    run_id: str = field(default_factory=new_run_id)
    internal: bool = True
    runner: DockerRunner = field(default=default_docker_runner, repr=False)
    cmd_timeout_s: float = field(default_factory=lambda: _defaults_from_config()[4])
    name: str = field(init=False)
    _created: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = f"{NAME_PREFIX}{self.run_id}"

    def __enter__(self) -> BenchNetwork:
        cmd = ["docker", "network", "create"]
        if self.internal:
            cmd.append("--internal")
        cmd.append(self.name)
        result = self.runner(cmd, timeout=self.cmd_timeout_s)
        if result.returncode != 0:
            raise_docker_failure(
                "docker network create failed",
                result.stderr.strip() or result.stdout,
            )
        self._created = True
        logger.info("created network %s (internal=%s)", self.name, self.internal)
        return self

    def __exit__(self, *exc: Any) -> None:
        if not self._created:
            return
        try:
            result = self.runner(
                ["docker", "network", "rm", self.name],
                timeout=self.cmd_timeout_s,
            )
            if result.returncode != 0:
                logger.warning(
                    "docker network rm %s failed: %s",
                    self.name,
                    result.stderr.strip() or result.stdout,
                )
        except (EngineError, HostEnvironmentError) as err:
            logger.warning("docker network rm %s error: %s", self.name, err)
        finally:
            self._created = False


def resolve_image_digest(
    image: str,
    *,
    runner: DockerRunner,
    cmd_timeout_s: float,
) -> str:
    """Record what we ran: pinned digest, else RepoDigests[0], else image Id."""
    pinned = extract_digest_from_image_ref(image)
    if pinned:
        return pinned

    # Prefer RepoDigests (registry-backed). Empty for never-pushed local builds.
    fmt = "{{json .RepoDigests}}"
    result = runner(
        ["docker", "inspect", "--format", fmt, image],
        timeout=cmd_timeout_s,
    )
    if result.returncode == 0:
        try:
            digests = json.loads(result.stdout.strip() or "[]")
        except json.JSONDecodeError:
            digests = []
        if digests:
            # e.g. ghcr.io/foo/bar@sha256:abc...
            ref = digests[0]
            if "@" in ref:
                return ref.rsplit("@", 1)[1]
            return ref

    id_result = runner(
        ["docker", "inspect", "--format", "{{.Id}}", image],
        timeout=cmd_timeout_s,
    )
    if id_result.returncode != 0:
        raise EngineError(
            f"cannot resolve digest for {image!r}: "
            f"{id_result.stderr.strip() or id_result.stdout}"
        )
    return normalize_image_id(id_result.stdout)


def container_ip_on_network(
    container_id: str,
    network_name: str,
    *,
    runner: DockerRunner,
    cmd_timeout_s: float,
) -> str:
    fmt = f'{{{{index .NetworkSettings.Networks "{network_name}" "IPAddress"}}}}'
    result = runner(
        ["docker", "inspect", "--format", fmt, container_id],
        timeout=cmd_timeout_s,
    )
    if result.returncode != 0:
        raise EngineError(
            f"docker inspect IP failed: {result.stderr.strip() or result.stdout}"
        )
    ip = result.stdout.strip()
    if not ip:
        raise EngineError(
            f"container {container_id[:12]} has no IP on network {network_name}"
        )
    return ip


def published_host_port(
    container_id: str,
    container_port: int,
    *,
    runner: DockerRunner,
    cmd_timeout_s: float,
) -> int:
    """Parse `docker port <id> <port>` → host port (random publish)."""
    result = runner(
        ["docker", "port", container_id, str(container_port)],
        timeout=cmd_timeout_s,
    )
    if result.returncode != 0:
        raise EngineError(
            f"docker port failed: {result.stderr.strip() or result.stdout}"
        )
    # e.g. "127.0.0.1:49153" or "0.0.0.0:49153\n[::]:49153"
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if ":" not in line:
        raise EngineError(f"unexpected docker port output: {result.stdout!r}")
    host_port = line.rsplit(":", 1)[-1].strip()
    try:
        return int(host_port)
    except ValueError as exc:
        raise EngineError(f"unexpected docker port output: {result.stdout!r}") from exc


def container_running(
    container_id: str,
    *,
    runner: DockerRunner,
    cmd_timeout_s: float,
) -> bool:
    """Return True iff Docker reports State.Running.

    Inspect/daemon failures raise ``EngineError`` — they must not be treated
    as "container exited" (that hides the real Docker error in the health loop).
    """
    result = runner(
        ["docker", "inspect", "--format", "{{.State.Running}}", container_id],
        timeout=cmd_timeout_s,
    )
    if result.returncode != 0:
        raise EngineError(
            f"docker inspect Running failed for {container_id[:12]}: "
            f"{result.stderr.strip() or result.stdout}"
        )
    return result.stdout.strip().lower() == "true"


def fetch_container_logs(
    container_id: str,
    *,
    runner: DockerRunner,
    cmd_timeout_s: float,
    tail: int | None = None,
) -> str:
    cmd = ["docker", "logs"]
    if tail is not None:
        cmd.extend(["--tail", str(tail)])
    cmd.append(container_id)
    result = runner(cmd, timeout=cmd_timeout_s)
    # logs may write to stderr; combine both
    return (result.stdout or "") + (result.stderr or "")


def wait_until_healthy(
    base_url: str,
    *,
    timeout_s: float,
    poll_s: float,
    is_alive: Callable[[], bool] | None = None,
    path: str = "/v1/models",
) -> None:
    """Poll GET {base_url}{path} until 200 or timeout / process death.

    ``is_alive`` returns False when the engine process/container has exited
    (fail-fast — do not burn the full timeout budget).
    """
    url = base_url.rstrip("/") + path
    deadline = time.monotonic() + timeout_s
    last_err: str | None = None
    while time.monotonic() < deadline:
        if is_alive is not None and not is_alive():
            raise EngineError(
                f"engine died before becoming healthy (last error: {last_err})"
            )
        try:
            with urlopen(url, timeout=min(poll_s, 5.0)) as resp:
                if 200 <= getattr(resp, "status", 200) < 300:
                    return
                last_err = f"HTTP {getattr(resp, 'status', '?')}"
        except URLError as exc:
            last_err = str(exc.reason if hasattr(exc, "reason") else exc)
        except Exception as exc:  # noqa: BLE001 — retry until timeout
            last_err = str(exc)
        time.sleep(poll_s)
    raise EngineError(
        f"health check timed out after {timeout_s}s for {url} (last: {last_err})"
    )


@dataclass
class EngineContainer:
    """Start an engine container, wait until healthy, tear down with logs."""

    spec: EngineSpec
    network: BenchNetwork
    role: str = "engine"
    gpu_count: int = 0
    weights_dir: Path | None = None
    port: int | None = None
    publish_port: bool = False
    pull: bool = True
    health_timeout_s: float | None = None
    health_poll_s: float | None = None
    logs_dir: Path | None = None
    env_extra: dict[str, str] | None = None
    runner: DockerRunner | None = None
    pull_timeout_s: float | None = None
    cmd_timeout_s: float | None = None
    # Progress only; a raising hook must not fail a healthy engine.
    on_ready: Callable[[], None] | None = None
    # Mount the host engine compile cache into this container, read-write.
    # A round asks for it on its baseline engine and never on a candidate, so
    # every candidate starts from the same cache state and whatever it
    # compiles dies with the container. Off unless a caller asks for it.
    mount_engine_cache: bool = False

    _container_id: str | None = field(default=None, init=False, repr=False)
    _container_name: str = field(default="", init=False, repr=False)
    _env_file: Path | None = field(default=None, init=False, repr=False)
    _handle: EngineHandle | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        h_to, h_poll, eng_port, pull_to, cmd_to = _defaults_from_config()
        if self.port is None:
            self.port = eng_port
        if self.health_timeout_s is None:
            self.health_timeout_s = h_to
        if self.health_poll_s is None:
            self.health_poll_s = h_poll
        if self.pull_timeout_s is None:
            self.pull_timeout_s = pull_to
        if self.cmd_timeout_s is None:
            self.cmd_timeout_s = cmd_to
        if self.runner is None:
            self.runner = self.network.runner
        safe_role = re.sub(r"[^a-zA-Z0-9_.-]", "-", self.role) or "engine"
        self._container_name = f"{NAME_PREFIX}{self.network.run_id}-{safe_role}"

    def _teardown(self) -> None:
        """Capture logs then ``docker rm -f``. Safe to call from ``__enter__`` failure.

        Python does not invoke ``__exit__`` when ``__enter__`` raises, so any
        failure after ``docker run`` must call this explicitly.
        """
        assert self.runner is not None
        assert self.cmd_timeout_s is not None
        cid = self._container_id
        try:
            if cid:
                if self.logs_dir is not None:
                    try:
                        self.logs_dir.mkdir(parents=True, exist_ok=True)
                        logs = fetch_container_logs(
                            cid,
                            runner=self.runner,
                            cmd_timeout_s=self.cmd_timeout_s,
                        )
                        log_path = self.logs_dir / f"{self._container_name}.log"
                        log_path.write_text(logs, encoding="utf-8")
                    except Exception as log_err:  # noqa: BLE001 — never block rm
                        logger.warning(
                            "failed to capture logs for %s: %s",
                            self._container_name,
                            log_err,
                        )
                try:
                    rm = self.runner(
                        ["docker", "rm", "-f", cid],
                        timeout=self.cmd_timeout_s,
                    )
                    if rm.returncode != 0:
                        logger.warning(
                            "docker rm -f %s failed: %s",
                            cid[:12],
                            rm.stderr.strip() or rm.stdout,
                        )
                except (EngineError, HostEnvironmentError) as rm_err:
                    logger.warning("docker rm -f %s error: %s", cid[:12], rm_err)
        finally:
            self._container_id = None
            if self._env_file is not None:
                try:
                    self._env_file.unlink(missing_ok=True)
                except OSError:
                    pass
                self._env_file = None

    def __enter__(self) -> EngineHandle:
        assert self.runner is not None
        assert self.port is not None
        assert self.health_timeout_s is not None
        assert self.health_poll_s is not None
        assert self.pull_timeout_s is not None
        assert self.cmd_timeout_s is not None

        if self.pull:
            pull_result = self.runner(
                ["docker", "pull", self.spec.image],
                timeout=self.pull_timeout_s,
            )
            if pull_result.returncode != 0:
                raise_docker_failure(
                    f"docker pull failed for {self.spec.image!r}",
                    pull_result.stderr.strip() or pull_result.stdout,
                )

        image_digest = resolve_image_digest(
            self.spec.image,
            runner=self.runner,
            cmd_timeout_s=self.cmd_timeout_s,
        )

        env = dict(self.spec.env)
        if self.env_extra:
            env.update(self.env_extra)

        # Pass env via --env-file so values never appear on the argv (or logs).
        if env:
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".env",
                prefix="pareton-bench-env-",
                delete=False,
                encoding="utf-8",
            )
            with tmp:
                for k, v in env.items():
                    # Docker env-file: KEY=VALUE; escape newlines
                    safe_v = str(v).replace("\n", "\\n")
                    tmp.write(f"{k}={safe_v}\n")
            self._env_file = Path(tmp.name)

        run_cmd: list[str] = [
            "docker",
            "run",
            "-d",
            "--name",
            self._container_name,
            "--network",
            self.network.name,
        ]
        if self.publish_port:
            # Random host port — discover via `docker port` after start.
            run_cmd.extend(["-p", f"127.0.0.1::{self.port}"])
        if self.gpu_count > 0:
            run_cmd.extend(["--gpus", str(self.gpu_count)])
            # Default docker /dev/shm is 64MB; NCCL TP>1 dies with
            # "unhandled system error" without host IPC + a larger shm.
            run_cmd.extend(["--ipc", "host", "--shm-size", "16g"])
        engine_cache = os.environ.get("PARETON_BENCH_ENGINE_CACHE_DIR", "").strip()
        if self.mount_engine_cache and engine_cache:
            cache_path = Path(engine_cache)
            cache_path.mkdir(parents=True, exist_ok=True)
            # TODO(PAR-81): the container-side path belongs in the
            # campaign's engine profile as a ``cache_dir`` key. Until then it
            # is hardcoded to the SGLang cache location.
            run_cmd.extend(["-v", f"{cache_path.resolve()}:/root/.cache/sglang"])
        if self.weights_dir is not None:
            run_cmd.extend(["-v", f"{self.weights_dir.resolve()}:/model:ro"])
        if self._env_file is not None:
            run_cmd.extend(["--env-file", str(self._env_file)])
        run_cmd.append(self.spec.image)
        run_cmd.extend(ensure_listen_args(self.spec.serve_args, self.port))

        # From docker run onward, any failure — including KeyboardInterrupt —
        # must tear down: __exit__ is NOT called when __enter__ raises.
        try:
            run_result = self.runner(run_cmd, timeout=self.cmd_timeout_s)
            if run_result.returncode != 0:
                raise_docker_failure(
                    "docker run failed",
                    run_result.stderr.strip() or run_result.stdout,
                )
            container_id = run_result.stdout.strip()
            if not container_id:
                raise EngineError("docker run returned empty container id")
            self._container_id = container_id

            if self.publish_port:
                host_port = published_host_port(
                    container_id,
                    self.port,
                    runner=self.runner,
                    cmd_timeout_s=self.cmd_timeout_s,
                )
                base_url = f"http://127.0.0.1:{host_port}"
            else:
                ip = container_ip_on_network(
                    container_id,
                    self.network.name,
                    runner=self.runner,
                    cmd_timeout_s=self.cmd_timeout_s,
                )
                base_url = f"http://{ip}:{self.port}"

            def _alive() -> bool:
                assert self.runner is not None and self.cmd_timeout_s is not None
                return container_running(
                    container_id,
                    runner=self.runner,
                    cmd_timeout_s=self.cmd_timeout_s,
                )

            try:
                wait_until_healthy(
                    base_url,
                    timeout_s=self.health_timeout_s,
                    poll_s=self.health_poll_s,
                    is_alive=_alive,
                )
            except EngineError as err:
                try:
                    tail = fetch_container_logs(
                        container_id,
                        runner=self.runner,
                        cmd_timeout_s=self.cmd_timeout_s,
                        tail=200,
                    )
                except EngineError:
                    tail = ""
                raise EngineError(
                    f"{err}; container log tail:\n{tail[-4000:]}"
                ) from err
        except BaseException:
            self._teardown()
            raise

        self._handle = EngineHandle(
            base_url=base_url,
            container_id=container_id,
            container_name=self._container_name,
            image_digest=image_digest,
            image=self.spec.image,
        )
        logger.info(
            "engine healthy name=%s digest=%s url=%s",
            self._container_name,
            image_digest,
            base_url,
        )
        if self.on_ready is not None:
            try:
                self.on_ready()
            except Exception as exc:  # noqa: BLE001 - progress must not fail a run
                logger.warning("on_ready hook failed: %s", exc)
        return self._handle

    def __exit__(self, *exc: Any) -> None:
        self._teardown()
