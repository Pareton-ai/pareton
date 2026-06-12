#!/usr/bin/env python3
"""Create and deploy a Targon GPU rental workload with a fresh volume + SSH keys."""

import os
import sys
import time
import uuid

import requests

from shared import SSH_KEYS, load_env_dict

WORKLOAD_NAME = "test-cacheon"  # TODO: replace this with your workload name
IMAGE = "ghcr.io/manifold-inc/ubuntu-systemd-docker:v3"
VOLUME_MOUNT = "/workspace"
VOLUME_SIZE_MB = 1_048_576  # 1 TiB
VOLUME_RESOURCE_NAME = "storage-rentals"

# Targon has no inventory fallback API; pick exactly one tier below.
TARGON_GPU = "H200"  # TODO: replace with one of "H100", "H200", "B200"

TARGON_RESOURCE_BY_GPU = {
    "H100": "h100-small",
    "H200": "h200-small",
    "B200": "b200-small",
}

BASE = "https://api.targon.com/tha/v2"


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {os.environ['TARGON_API_KEY']}",
        "Content-Type": "application/json",
    }


def _req(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{BASE}{path}"
    resp = requests.request(method, url, headers=_headers(), **kwargs)
    if not resp.ok:
        print(f"  {method} {path} -> {resp.status_code}", file=sys.stderr)
        print(f"    {resp.text}", file=sys.stderr)
        sys.exit(1)
    return resp


def ensure_ssh_keys() -> list[str]:
    """Register any missing SSH keys and return all UIDs."""
    resp = _req("GET", "/ssh-keys?limit=100")
    existing = {k["name"]: k["uid"] for k in resp.json().get("items", [])}

    uids: list[str] = []
    for name, pubkey in SSH_KEYS.items():
        if name in existing:
            uid = existing[name]
            print(f"  SSH key '{name}' already exists -> {uid}")
        else:
            r = _req("POST", "/ssh-keys", json={"name": name, "ssh_key": pubkey})
            uid = r.json()["uid"]
            print(f"  SSH key '{name}' created -> {uid}")
        uids.append(uid)
    return uids


def create_volume() -> str:
    name = f"cacheon-manual-{uuid.uuid4().hex[:8]}"
    print(f"\nCreating volume '{name}' (1 TiB)...")
    resp = _req(
        "POST",
        "/volumes",
        json={
            "name": name,
            "size_in_mb": VOLUME_SIZE_MB,
            "resource_name": VOLUME_RESOURCE_NAME,
        },
    )
    volume_uid = resp.json()["uid"]
    print(f"  Volume registered -> {volume_uid}")
    return volume_uid


def wait_for_volume(volume_uid: str, interval: int = 10) -> None:
    print(f"\nWaiting for volume {volume_uid} to reach READY...")
    while True:
        resp = _req("GET", f"/volumes/{volume_uid}/state")
        status = resp.json().get("status", "UNKNOWN").upper()
        print(f"  status={status}")
        if status == "READY" or status == "REGISTERED":
            print(f"  Volume {volume_uid} is ready.")
            return
        if status in ("FAILED", "ERROR", "DELETING"):
            print(f"\n  Volume entered terminal state: {status}", file=sys.stderr)
            sys.exit(1)
        time.sleep(interval)


def create_workload(
    ssh_key_uids: list[str], volume_uid: str, pod_envs: list[dict]
) -> str:
    """Create a rental workload for the configured GPU tier."""
    resource = TARGON_RESOURCE_BY_GPU.get(TARGON_GPU)
    if not resource:
        allowed = ", ".join(sorted(TARGON_RESOURCE_BY_GPU))
        print(
            f"  ERROR: TARGON_GPU must be one of [{allowed}], got {TARGON_GPU!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    payload = {
        "name": WORKLOAD_NAME,
        "image": IMAGE,
        "resource_name": resource,
        "type": "RENTAL",
        "volumes": [{"uid": volume_uid, "mount_path": VOLUME_MOUNT}],
        "ssh_keys": ssh_key_uids,
        "envs": pod_envs,
    }
    print(f"\nCreating workload '{WORKLOAD_NAME}' ({TARGON_GPU} -> {resource})...")
    print(f"  Injecting {len(pod_envs)} env vars from .env")
    resp = _req("POST", "/workloads", json=payload)
    data = resp.json()
    uid = data["uid"]
    print(f"  Workload registered -> {uid}")
    print(f"  Cost: ${data.get('cost_per_hour', '?')}/hr")
    return uid


def deploy_workload(workload_uid: str) -> None:
    print(f"\nDeploying {workload_uid}...")
    resp = _req("POST", f"/workloads/{workload_uid}/deploy")
    state = resp.json().get("state", {})
    print(f"  Status: {state.get('status', 'UNKNOWN')}")


def wait_for_running(workload_uid: str, volume_uid: str, interval: int = 15) -> None:
    print("\nWaiting for workload to reach RUNNING...")
    while True:
        resp = _req("GET", f"/workloads/{workload_uid}/state")
        data = resp.json()
        status = data.get("status", "UNKNOWN").upper()
        print(f"  status={status}")
        if status == "RUNNING":
            print("\n  Workload is RUNNING.")
            print(f"  ssh {workload_uid}@ssh.deployments.targon.com")
            print(
                f"\n  Volume {volume_uid} persists until you delete it in the Targon dashboard "
                "or via POST /tha/v2/volumes/{uid}/delete."
            )
            return
        if status in ("FAILED", "TERMINATED", "ERROR"):
            print(f"\n  Workload entered terminal state: {status}", file=sys.stderr)
            sys.exit(1)
        time.sleep(interval)


def main() -> None:
    if not os.getenv("TARGON_API_KEY"):
        print("ERROR: TARGON_API_KEY not found in environment", file=sys.stderr)
        sys.exit(1)

    print("=== Targon GPU Pod Creator ===\n")

    print("1) Ensuring SSH keys...")
    ssh_uids = ensure_ssh_keys()

    print("\n2) Loading .env for pod injection...")
    env_dict = load_env_dict()
    pod_envs = [{"name": k, "value": v} for k, v in env_dict.items()]
    print(f"  {len(pod_envs)} env vars loaded (values hidden)")

    print("\n3) Creating block volume...")
    volume_uid = create_volume()
    wait_for_volume(volume_uid)

    print("\n4) Creating rental workload...")
    wid = create_workload(ssh_uids, volume_uid, pod_envs)

    print("\n5) Deploying...")
    deploy_workload(wid)

    print("\n6) Polling status...")
    wait_for_running(wid, volume_uid)

    print("\nDone.")


if __name__ == "__main__":
    main()
