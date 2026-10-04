"""Thin HTTP client for the backend API (the frontend never reads files or talks to the model directly).

The API base URL comes from the API_URL environment variable (default http://localhost:8000). Images are fetched
server-side as bytes, so the user's browser never needs to reach the API (it can't in a codespace).
"""
from __future__ import annotations

import os

import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = httpx.Timeout(8.0, connect=3.0)
RUN_API = "uvicorn backend.api:app --host 0.0.0.0 --port 8000"


class ApiUnavailable(RuntimeError):
    """The API could not be reached or did not answer in time."""


class ApiError(RuntimeError):
    """The API answered with an error status (404, 422, ...)."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def _get(path: str, params: dict | None = None) -> httpx.Response:
    try:
        return httpx.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
    except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
        raise ApiUnavailable(f"Cannot reach the API at {API_URL} ({type(e).__name__}).") from e


def get_json(path: str, params: dict | None = None) -> dict:
    r = _get(path, params)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise ApiError(r.status_code, str(detail))
    return r.json()


def get_bytes(url: str | None) -> bytes | None:
    """Image bytes for an API image URL ('/images/...'), or None if missing / unreachable."""
    if not url:
        return None
    try:
        r = _get(url)
    except ApiUnavailable:
        return None
    return r.content if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/") else None
