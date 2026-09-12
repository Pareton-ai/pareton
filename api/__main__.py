"""Run: python -m api"""

import logging
import threading

import uvicorn

from observability import probe as obs_probe

if __name__ == "__main__":
    # Lifecycle events (deployment probes) log at INFO; without this the
    # root logger's default WARNING threshold silently drops them.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    # Read-only deployment-probe poller (stage-2 spec 7.2): a daemon thread
    # so a re-verify against the running API never restarts it.
    threading.Thread(
        target=obs_probe.run_probe_loop, args=("pareton-api",), daemon=True
    ).start()
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
