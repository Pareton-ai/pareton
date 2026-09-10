"""Exercise the real Compose services with isolated Postgres and mock workers.

Build the runtime image first. No production env, chain, cloud or Axiom access.
The temporary project's containers and volumes are removed on exit.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="pareton-runtime:local")
    args = parser.parse_args()
    project = "pareton-smoke-" + uuid.uuid4().hex[:8]
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PARETON_", "COMPOSE_"))
    }
    env.update(PARETON_AXIOM_TOKEN="test", PARETON_ENV_FILE=str(ROOT / ".env.example"))

    def run(*command, **kwargs):
        return subprocess.run(
            command, env=env, text=True, capture_output=True, check=True, **kwargs
        ).stdout.strip()

    config = json.loads(
        run(
            "docker",
            "compose",
            "--env-file",
            str(ROOT / ".env.example"),
            "-f",
            str(ROOT / "compose.yaml"),
            "config",
            "--format",
            "json",
        )
    )
    selected = ("api", "worker", "round-worker", "vector")
    config["services"] = {key: config["services"][key] for key in selected}
    config["name"] = project
    config["networks"] = {"default": {}}
    config["volumes"] = {name: {} for name in ("db", "logs", "work", "vector-data")}
    common = {
        "PARETON_DATABASE_URL": "postgresql://test:test@postgres/test",
        "PARETON_NETWORK": "test",
        "PARETON_NETUID": "543",
        "PARETON_ALLOW_MOCK_BENCH": "1",
        "PARETON_WEIGHTS_ENABLED": "false",
        "PARETON_REQUIRE_BUILDER": "0",
        "PARETON_POLL_INTERVAL_S": "1",
        "PARETON_WORK_DIR": "/var/lib/pareton/work",
        "PARETON_BUILD_LOG_DIR": "/var/log/pareton/builds",
        "PARETON_HEALTH_FILE": "/tmp/worker-heartbeat",
    }
    for name in ("api", "worker", "round-worker"):
        service = config["services"][name]
        service.pop("build", None)
        service.pop("env_file", None)
        service["environment"] = common
        service["image"] = args.image
        service["restart"] = "no"
        service["stop_grace_period"] = "10s"
        service["volumes"] = [
            "work:/var/lib/pareton/work",
            "logs:/var/log/pareton/builds",
        ]
        service["depends_on"]["postgres"] = {"condition": "service_healthy"}
        service["healthcheck"].update(interval="1s", start_period="1s", retries=30)
    config["services"]["worker"]["command"] += ["--mock-build"]
    config["services"]["round-worker"]["command"] += ["--mock-bench"]
    config["services"]["api"]["ports"] = ["127.0.0.1::8000"]
    config["services"]["postgres"] = {
        "image": "postgres:16-alpine",
        "environment": {
            "POSTGRES_USER": "test",
            "POSTGRES_PASSWORD": "test",
            "POSTGRES_DB": "test",
        },
        "volumes": [
            "db:/var/lib/postgresql/data",
            f"{ROOT}/db/schema.sql:/docker-entrypoint-initdb.d/schema.sql:ro",
        ],
        "healthcheck": {
            "test": ["CMD-SHELL", "pg_isready -U test -d test"],
            "interval": "1s",
            "timeout": "5s",
            "retries": 60,
        },
    }
    with tempfile.TemporaryDirectory(prefix="pareton-compose-") as directory:
        vector_config = Path(directory) / "vector.toml"
        # Exercise the production source/transforms, with a local console sink.
        pipeline = (
            (ROOT / "ops/vector/vector.toml").read_text().split("[sinks.axiom]")[0]
        )
        vector_config.write_text(
            pipeline
            + '\n[sinks.console]\ntype="console"\ninputs=["parse_lifecycle"]\nencoding.codec="json"\n'
        )
        vector = config["services"]["vector"]
        vector["environment"] = {"PARETON_COMPOSE_PROJECT": project}
        vector["volumes"] = [
            "/var/run/docker.sock:/var/run/docker.sock",
            f"{vector_config}:/etc/vector/vector.toml:ro",
            "vector-data:/var/lib/vector",
        ]
        compose_file = Path(directory) / "compose.json"
        compose_file.write_text(json.dumps(config))

        def compose(*command, **kwargs):
            return run(
                "docker",
                "compose",
                "-p",
                project,
                "-f",
                str(compose_file),
                *command,
                **kwargs,
            )

        def query(sql):
            return compose(
                "exec",
                "-T",
                "postgres",
                "psql",
                "-U",
                "test",
                "-d",
                "test",
                "-Atc",
                sql,
            )

        def check_api():
            port = compose("port", "api", "8000")
            with urlopen(f"http://{port}/health", timeout=10) as response:
                assert json.load(response)["ok"]
            with urlopen(f"http://{port}/v1/campaigns", timeout=10) as response:
                assert response.status == 200

        try:
            compose("up", "-d", "--no-build", "--wait", "--wait-timeout", "120")
            check_api()
            compose(
                "exec",
                "-T",
                "worker",
                "python",
                "-c",
                "from pathlib import Path; Path('/var/log/pareton/builds/smoke').write_text('persisted')",
            )
            query(
                "CREATE TABLE compose_smoke (value text); INSERT INTO compose_smoke VALUES ('persisted')"
            )
            deadline = time.monotonic() + 30
            while True:
                logs = compose("logs", "--no-color", "vector")
                if all(
                    unit in logs
                    for unit in (
                        "pareton-worker.service",
                        "pareton-round-worker.service",
                    )
                ):
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("Vector did not collect both worker identities")
                time.sleep(0.5)
            compose("down")
            compose("up", "-d", "--no-build", "--wait", "--wait-timeout", "120")
            check_api()
            assert query("SELECT value FROM compose_smoke") == "persisted"
            assert (
                compose("exec", "-T", "api", "cat", "/var/log/pareton/builds/smoke")
                == "persisted"
            )
            print(
                "Compose smoke passed: API/Postgres, worker heartbeats through Vector, shared logs, down/up persistence"
            )
        except Exception:
            print(compose("logs", "--no-color", "--tail", "80"))
            raise
        finally:
            # Only this randomly named disposable test project's volumes.
            compose("down", "--volumes", "--remove-orphans")


if __name__ == "__main__":
    main()
