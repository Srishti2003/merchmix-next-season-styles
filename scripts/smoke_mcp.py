"""Smoke test: launch each MCP server over stdio, list its tools and call every tool once.

Run from the repo root: python scripts/smoke_mcp.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

ROOT = Path(__file__).resolve().parents[1]
CODE = "0751471"
CALLS: dict[str, list[tuple[str, dict]]] = {
    "retail_data.py": [
        ("get_style_attributes", {"product_code": CODE}),
        ("get_sales_curve", {"product_code": CODE, "weeks": 6}),
        ("get_reference_images", {"product_code": CODE}),
        ("season_summary", {"season": "summer", "year": 2020}),
    ],
    "forecast.py": [
        ("predict_top_k", {"cutoff": "2020-09-22", "k": 3, "diversify": True}),
        ("explain_style", {"product_code": CODE, "cutoff": "2020-09-22"}),
        ("write_evidence", {"cutoff": "2020-09-22", "product_codes": [CODE]}),  # rank-1 winner: rewrites identical files
        ("evaluation_report", {}),
    ],
}
REF = str(ROOT / "outputs" / "refs" / CODE / f"{CODE}001.jpg")
CALLS["image_gen.py"] = [
    ("list_concepts", {"product_code": CODE}),
    ("generate_concept", {"product_code": CODE, "reference_image": REF,
                          "prompt": "Redesign as a next-season trouser: keep the tapered silhouette, change fabric to check."}),
    ("similarity_score", {"image_a": REF, "image_b": lambda last: last["path"]}),
    ("novelty_check", {"product_code": CODE, "concept_path": lambda last: last["path"]}),
    ("compose_board", {"entries": lambda last: [{"product_code": CODE, "ref_path": REF,
                                                  "concept_path": last["concept_path"], "why_it_won": "smoke test",
                                                  "what_changed": "smoke test"}]}),
]
# The image server runs on the mock generator and writes to a scratch evidence dir in the smoke test.
ENV = {"image_gen.py": {"IMAGE_BACKEND": "mock",
                        "HM_EVIDENCE_DIR": str(ROOT / "outputs" / "mock_run" / "evidence")}}
ERROR_CALLS = [("retail_data.py", "get_style_attributes", {"product_code": "9999999"}),
               ("forecast.py", "predict_top_k", {"cutoff": "2021-01-01"}),
               ("image_gen.py", "novelty_check", {"product_code": CODE, "concept_path": "missing.png"})]


def short(obj: object, n: int = 300) -> str:
    """Compact one-line JSON preview."""
    s = json.dumps(obj, default=str, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + " …"


async def run_server(script: str, calls: list[tuple[str, dict]]) -> int:
    """Call every tool on one server; return the number of successful calls."""
    transport = PythonStdioTransport(ROOT / "mcp_servers" / script, python_cmd=sys.executable, cwd=str(ROOT),
                                     env={**os.environ, **ENV.get(script, {})})
    ok = 0
    async with Client(transport, timeout=600) as client:
        tools = await client.list_tools()
        print(f"\n== {script}: tools = {[t.name for t in tools]}")
        assert {t.name for t in tools} == {c[0] for c in calls}, "tool list mismatch"
        last: dict = {}
        for name, args in calls:
            t = time.time()
            args = {k: (v(last) if callable(v) else v) for k, v in args.items()}  # chain earlier results
            res = await client.call_tool(name, args)
            payload = res.structured_content if res.structured_content is not None else res.data
            if isinstance(payload, dict):
                last.update(payload)
            print(f"  ✓ {name}({short(args, 80)}) [{time.time() - t:.1f}s] -> {short(payload)}")
            ok += 1
        for s, name, args in ERROR_CALLS:
            if s != script:
                continue
            res = await client.call_tool(name, args, raise_on_error=False)
            print(f"  ✓ error path {name}({short(args, 60)}) -> {res.content[0].text[:160]}")
    return ok


async def main() -> None:
    """Run all servers sequentially and report."""
    total = 0
    for script, calls in CALLS.items():
        total += await run_server(script, calls)
    expected = sum(len(c) for c in CALLS.values())
    print(f"\n{total}/{expected} tools called successfully")
    if total != expected:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
