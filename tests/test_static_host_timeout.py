"""Real Linux process/lock regression; no Docker, GPU, network, or database."""

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gpu.static_host import (
    HostBusyError,
    bounded_bench_command,
    host_lock,
    reap_idle_containers,
)


@unittest.skipUnless(
    sys.platform == "linux"
    and all(shutil.which(tool) for tool in ("nohup", "setsid", "timeout", "flock")),
    "requires Linux coreutils and util-linux",
)
class StaticTimeoutTest(unittest.TestCase):
    def test_hangup_preserves_deadline_escalation_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = root / "host.lock"
            ready = root / "ready.json"
            term = root / "term"
            payload = (
                "import json, os, signal, time; from pathlib import Path; "
                f"signal.signal(signal.SIGTERM, lambda *a: Path({str(term)!r}).touch()); "
                f"Path({str(ready)!r}).write_text(json.dumps([os.getpid(), os.getsid(0), os.getpgrp()])); "
                "print('harness ready', flush=True); time.sleep(120)"
            )
            with patch("gpu.static_host.REMOTE_LOCK", str(lock)):
                command = bounded_bench_command(
                    f"exec {shlex.quote(sys.executable)} -c {shlex.quote(payload)}",
                    output_dir=str(root / "output with spaces"),
                    timeout_s=2,
                )
            # Stand in for sshd's session: terminating it must not kill the timer.
            transport = subprocess.Popen(
                ["sh", "-c", f"sh -c {shlex.quote(command)} & wait"],
                start_new_session=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            process_group = None
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(ready.exists(), "test harness never started")
                pid, session, process_group = json.loads(ready.read_text())
                self.assertNotEqual(session, transport.pid)
                with self.assertRaises(HostBusyError):
                    with host_lock(lock):
                        pass
                os.killpg(transport.pid, signal.SIGHUP)
                transport.wait(timeout=5)
                for stream in (transport.stdin, transport.stdout, transport.stderr):
                    stream.close()
                deadline = time.monotonic() + 5
                while not term.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(term.exists(), "remote timeout never sent TERM")
                with self.assertRaises(HostBusyError):
                    with host_lock(lock):
                        pass  # TERM-resistant harness must still hold the lock.
                deadline = time.monotonic() + 40
                while time.monotonic() < deadline:
                    try:
                        with host_lock(lock):
                            break
                    except HostBusyError:
                        time.sleep(0.05)
                else:
                    self.fail("KILL escalation failed to release the remote lock")
                stat = Path(f"/proc/{pid}/stat")
                self.assertTrue(not stat.exists() or stat.read_text().split()[2] == "Z")
                self.assertEqual(
                    reap_idle_containers(
                        lock_path=lock,
                        docker=lambda *a: SimpleNamespace(stdout=""),
                        verify=lambda: None,
                    )["status"],
                    "cleaned",
                )
                self.assertIn(
                    "harness ready",
                    (root / "output with spaces/supervisor.log").read_text(),
                )
            finally:
                if process_group is not None:
                    try:
                        os.killpg(process_group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if transport.poll() is None:
                    os.killpg(transport.pid, signal.SIGKILL)
                transport.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
