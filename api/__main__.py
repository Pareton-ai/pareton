"""Run: python -m api"""

import uvicorn

if __name__ == "__main__":
    # Caddy is the public entry point; never expose the backend directly.
    uvicorn.run("api.server:app", host="127.0.0.1", port=8000, reload=False)
