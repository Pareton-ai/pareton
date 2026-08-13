"""CLI: python -m gpu ..."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from gpu.errors import DestroyError, GpuError, ProvisionError
from gpu.orchestrate import (
    destroy_pod,
    discover_calib_requests,
    provision_pod,
    run_bench_on_pod,
)
from gpu.providers import get_provider
from gpu.reap import reap
from gpu.registry import PodRegistry, parse_pod_name
from gpu.ssh import exec as ssh_exec
from gpu.types import Pod, PodSpec, SshTarget


def _defaults_from_config() -> tuple[float, int]:
    try:
        import config as _cfg

        return (
            float(getattr(_cfg, "GPU_TTL_HOURS", 2.0)),
            int(getattr(_cfg, "GPU_MAX_HOURLY_CENTS", 1000)),
        )
    except Exception:  # noqa: BLE001
        return 2.0, 1000


def _add_provision_flags(p: argparse.ArgumentParser) -> None:
    ttl, max_cents = _defaults_from_config()
    p.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "targon", "shadeform", "lium", "static_ssh"],
        help="auto → PARETON_GPU_PROVIDER (default lium)",
    )
    p.add_argument("--gpu-type", default=None)
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--max-hourly-cents", type=int, default=max_cents)
    p.add_argument("--ttl-hours", type=float, default=ttl)
    p.add_argument(
        "--force",
        action="store_true",
        help="Bypass single-flight (allow rent while another managed pod exists)",
    )
    p.add_argument(
        "--manual",
        action="store_true",
        help=(
            "Print dashboard rent instructions and poll until a workload with the "
            "encoded name is RUNNING (Targon workaround when API provision hangs)"
        ),
    )


def _spec_from_args(args: argparse.Namespace) -> PodSpec:
    return PodSpec(
        gpu_count=args.gpu_count,
        gpu_type=args.gpu_type,
        max_hourly_cents=args.max_hourly_cents,
        ttl_hours=args.ttl_hours,
        provider=args.provider,
        force=bool(getattr(args, "force", False)),
        manual=bool(getattr(args, "manual", False)),
    )


def _pod_from_registry(entry) -> Pod:
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


def cmd_provision(args: argparse.Namespace) -> int:
    try:
        pod = provision_pod(_spec_from_args(args))
    except (ProvisionError, GpuError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"name={pod.name}")
    print(f"pod_id={pod.pod_id}")
    print(f"provider={pod.provider}")
    print(f"hourly_cents={pod.hourly_price_cents}")
    print(f"ssh=ssh -i {pod.key_path} -p {pod.ssh.port} {pod.ssh.user}@{pod.ssh.host}")
    if pod.raw.get("volume_uid"):
        print(f"volume_uid={pod.raw['volume_uid']}")
    return 0


def cmd_destroy(args: argparse.Namespace) -> int:
    registry = PodRegistry()
    entry = registry.get(args.pod_name)
    if entry is None:
        print(f"error: unknown pod {args.pod_name!r} in registry", file=sys.stderr)
        return 2
    pod = _pod_from_registry(entry)
    try:
        destroy_pod(pod, registry=registry)
    except DestroyError as exc:
        print(f"error: destroy failed: {exc}", file=sys.stderr)
        return 2
    print(f"destroyed {pod.name}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    del args
    registry = PodRegistry()
    now = datetime.now(timezone.utc)
    rows = []
    for e in registry.list():
        parsed = parse_pod_name(e.name)
        remain = ""
        if parsed:
            _c, _t, deadline = parsed
            secs = (deadline - now).total_seconds()
            remain = f"{secs / 3600:.2f}h"
        rows.append(
            {
                "source": "registry",
                "name": e.name,
                "provider": e.provider,
                "pod_id": e.pod_id,
                "state": e.state,
                "ttl_remaining": remain,
                "volume_uid": e.volume_uid,
            }
        )
    for pname in ("targon", "shadeform", "lium"):
        try:
            provider = get_provider(pname, state_dir=registry.state_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"# skip {pname}: {exc}", file=sys.stderr)
            continue
        try:
            for pod in provider.list_pods():
                parsed = parse_pod_name(pod.name)
                remain = ""
                if parsed:
                    _c, _t, deadline = parsed
                    remain = f"{(deadline - now).total_seconds() / 3600:.2f}h"
                rows.append(
                    {
                        "source": "provider",
                        "name": pod.name,
                        "provider": pod.provider,
                        "pod_id": pod.pod_id,
                        "state": "remote",
                        "ttl_remaining": remain,
                        "volume_uid": (pod.raw or {}).get("volume_uid", ""),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            print(f"# list_pods {pname} failed: {exc}", file=sys.stderr)
    print(json.dumps(rows, indent=2))
    return 0


def cmd_reap(args: argparse.Namespace) -> int:
    actions = reap(dry_run=args.dry_run)
    failed = False
    for a in actions:
        flag = "DRY" if a.dry_run else ("OK" if a.destroyed else "FAIL")
        if not a.dry_run and not a.destroyed:
            failed = True
        print(f"{flag} {a.kind} {a.provider} {a.name} id={a.id} {a.detail}")
    return 1 if failed else 0


def cmd_bench(args: argparse.Namespace) -> int:
    request = getattr(args, "request", None)
    requests_dir = getattr(args, "requests_dir", None)
    if bool(request) == bool(requests_dir):
        print(
            "error: provide exactly one of --request or --requests-dir",
            file=sys.stderr,
        )
        return 2
    if requests_dir is not None and int(args.repetitions) != 1:
        print(
            "error: --requests-dir is incompatible with --repetitions",
            file=sys.stderr,
        )
        return 2
    try:
        paths = (
            discover_calib_requests(Path(requests_dir))
            if requests_dir is not None
            else None
        )
        return run_bench_on_pod(
            _spec_from_args(args),
            request_path=Path(request) if request is not None else None,
            request_paths=paths,
            output_dir=Path(args.output_dir),
            mock_engine=bool(args.mock_engine),
            repetitions=int(args.repetitions),
            pod_name=getattr(args, "pod_name", None),
            keep=bool(getattr(args, "keep", False)),
        )
    except (ProvisionError, GpuError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def cmd_exec(args: argparse.Namespace) -> int:
    registry = PodRegistry()
    entry = registry.get(args.pod_name)
    if entry is None:
        print(f"error: unknown pod {args.pod_name!r}", file=sys.stderr)
        return 2
    pod = _pod_from_registry(entry)
    cmd = " ".join(args.cmd)
    try:
        result = ssh_exec(pod, cmd, timeout_s=float(args.timeout), check=False)
    except GpuError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return int(result.exit_code)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m gpu",
        description="Pareton GPU pod rent / reap / remote bench",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_prov = sub.add_parser("provision", help="Rent a GPU pod")
    _add_provision_flags(p_prov)
    p_prov.set_defaults(func=cmd_provision)

    p_des = sub.add_parser("destroy", help="Destroy a registry pod")
    p_des.add_argument("pod_name")
    p_des.set_defaults(func=cmd_destroy)

    p_list = sub.add_parser("list", help="List registry + provider pods")
    p_list.set_defaults(func=cmd_list)

    p_reap = sub.add_parser("reap", help="Destroy expired pods/volumes")
    p_reap.add_argument("--dry-run", action="store_true")
    p_reap.set_defaults(func=cmd_reap)

    p_bench = sub.add_parser(
        "bench",
        help="Run bench request(s) on a GPU pod (default: provision, run, destroy)",
    )
    _add_provision_flags(p_bench)
    p_bench.add_argument("--request", type=Path, default=None)
    p_bench.add_argument(
        "--requests-dir",
        type=Path,
        default=None,
        help="Run each sample-*/bench_request.json into --output-dir/run-00N",
    )
    p_bench.add_argument("--output-dir", required=True, type=Path)
    p_bench.add_argument("--mock-engine", action="store_true")
    p_bench.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Replay one --request N times on one pod; incompatible with --requests-dir",
    )
    p_bench.add_argument(
        "--pod",
        dest="pod_name",
        default=None,
        help="Reuse this registry pod instead of provisioning",
    )
    p_bench.add_argument(
        "--keep",
        action="store_true",
        help="Do not destroy the pod when the command exits",
    )
    p_bench.set_defaults(func=cmd_bench)

    p_exec = sub.add_parser("exec", help="Run a command on a registry pod")
    p_exec.add_argument("pod_name")
    p_exec.add_argument("--timeout", type=float, default=600.0)
    p_exec.add_argument("cmd", nargs=argparse.REMAINDER)
    p_exec.set_defaults(func=cmd_exec)

    return p


def main(argv: list[str] | None = None) -> int:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "exec":
        # Allow `exec name -- cmd` form; strip leading --
        if args.cmd and args.cmd[0] == "--":
            args.cmd = args.cmd[1:]
        if not args.cmd:
            parser.error("exec requires a command after --")
    return int(args.func(args))
