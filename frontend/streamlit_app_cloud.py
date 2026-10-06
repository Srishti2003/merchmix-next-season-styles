"""Streamlit Community Cloud entrypoint: starts the FastAPI backend inside the app container, then runs app.py.

  streamlit run frontend/streamlit_app_cloud.py                   # no API_URL: starts the API on 127.0.0.1:8000
  MERCHMIX_API_PORT=7871 streamlit run frontend/streamlit_app_cloud.py   # same, on another port
  API_URL=http://localhost:8000 streamlit run frontend/streamlit_app_cloud.py   # use an API that is already running

The API runs as a subprocess from the repo root (once per container; Streamlit reruns reuse it) and app.py still
talks to it only over HTTP.
"""
from __future__ import annotations

import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

import httpx
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
APP = Path(__file__).resolve().parent / "app.py"
STARTUP_TIMEOUT = 60


def _healthy(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


@st.cache_resource(show_spinner=False)
def start_api(port: int) -> subprocess.Popen | None:
    """Start uvicorn once per process and wait for /health. Returns None if an API was already answering."""
    url = f"http://127.0.0.1:{port}"
    if _healthy(url):
        return None
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.api:app", "--host", "127.0.0.1",
                             "--port", str(port)], cwd=ROOT)
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"The API process exited with code {proc.returncode} during startup.")
        if _healthy(url):
            return proc
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError(f"The API did not answer {url}/health within {STARTUP_TIMEOUT} s.")


if not os.getenv("API_URL"):
    api_port = int(os.getenv("MERCHMIX_API_PORT", "8000"))
    start_api(api_port)
    os.environ["API_URL"] = f"http://127.0.0.1:{api_port}"  # read by frontend/api_client.py at import

runpy.run_path(str(APP), run_name="__main__")
