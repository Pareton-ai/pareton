"""Exercise the repo's Vector validation/startup with synthetic local credentials.

Run in a disposable Linux container with Vector 0.57.0 and no external network.
Only the Axiom endpoint, source and data directory are replaced; the token
expression and startup arguments come from the production files.
"""

import argparse
import importlib.util
import os
import shlex
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def main():
    ops = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("sync_config", ops / "sync-config.py")
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    print(subprocess.check_output(["vector", "--version"], text=True).strip())
    authorizations = []

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            auth = self.headers.get("Authorization")
            authorizations.append(auth)
            self.send_response(200 if auth == "Bearer synthetic-good-token" else 401)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_):
            pass

    with HTTPServer(("127.0.0.1", 0), Receiver) as receiver:
        threading.Thread(target=receiver.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory(prefix="pareton-vector-auth-") as tmp:
                root = Path(tmp)
                env_file = root / "synthetic.env"
                env_file.write_text("PARETON_AXIOM_TOKEN=synthetic-good-token\n")
                # Preserve the real sink/token configuration, replacing the source
                # with stdin and the destination with our local receiver.
                sink = (
                    (ops / "vector/vector.toml")
                    .read_text()
                    .split("[sinks.axiom]", 1)[1]
                )
                config = (
                    f'data_dir = "{tmp}"\n'
                    '[sources.parse_lifecycle]\ntype = "stdin"\n'
                    "[sinks.axiom]\n"
                    f'url = "http://127.0.0.1:{receiver.server_port}"\n'
                    'compression = "none"\nbatch.max_events = 1\n' + sink
                )
                # A real expansion is necessary for candidate validation to find
                # this directory. This catches a missing flag in the sync command.
                os.environ["PARETON_VECTOR_TEST_DATA"] = tmp
                candidate = config.replace(tmp, "${PARETON_VECTOR_TEST_DATA}", 1)
                failures = []
                try:
                    sync.validate_candidates(
                        argparse.Namespace(
                            skip_validation=False, env_file=str(env_file)
                        ),
                        [
                            (
                                sync.Entry(
                                    "ops/vector/vector.toml",
                                    "/etc/vector/vector.toml",
                                    0o600,
                                ),
                                candidate.encode(),
                            )
                        ],
                    )
                    print(
                        "PASS candidate validation interpolates environment variables"
                    )
                except sync.Fail as error:
                    failures.append(f"candidate validation: {error}")

                config_path = root / "vector.toml"
                config_path.write_text(config)
                unit = (ops / "vector/vector.service").read_text()
                command = shlex.split(
                    next(
                        line.removeprefix("ExecStart=")
                        for line in unit.splitlines()
                        if line.startswith("ExecStart=")
                    )
                )
                command[command.index("--config") + 1] = str(config_path)
                result = subprocess.run(
                    command,
                    input="synthetic-event\n",
                    env={
                        "PATH": os.environ["PATH"],
                        "PARETON_AXIOM_TOKEN": "synthetic-good-token",
                    },
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=20,
                )
                if result.returncode != 0 or authorizations != [
                    "Bearer synthetic-good-token"
                ]:
                    failures.append(
                        f"startup authentication: exit={result.returncode}, requests={authorizations!r}"
                    )
                else:
                    print("PASS unit startup sends the expanded synthetic token")
                assert not failures, "\n".join(failures)
        finally:
            receiver.shutdown()


if __name__ == "__main__":
    main()
