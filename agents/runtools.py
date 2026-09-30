"""In-process MCP server 'run': deterministic bookkeeping tools for one orchestrator run.

The LLM agents decide; these tools make sure what they decide is validated and recorded exactly:
- validate_brief  : the style-dna-brief skill's validator (skills_lib.style_dna) + saves brief.json
- record_critique : appends the critic's verdict + CLIP scores to critic.jsonl
- write_lineage   : assembles lineage.json (forecast → brief → prompt → concept → scores) from the files
Everything is written under outputs/evidence/<code>/ and stamped with the run id.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

import config
from image import generate
from skills_lib import style_dna

RUN: dict[str, Any] = {"run_id": None, "cutoff": None}  # set by the orchestrator before the session starts


def _now() -> str:
    """UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ok(obj: Any) -> dict:
    """MCP text result with JSON payload."""
    return {"content": [{"type": "text", "text": json.dumps(obj, ensure_ascii=False, default=str)}]}


def _err(msg: str) -> dict:
    """MCP error result."""
    return {"content": [{"type": "text", "text": msg}], "is_error": True}


def _dir(code: str) -> Path:
    """Evidence directory for a style (must already hold forecast.json)."""
    return config.EVIDENCE_DIR / str(code).zfill(7)


def _jsonl(path: Path) -> list[dict]:
    """Read a .jsonl file (empty list if missing)."""
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def do_validate_brief(product_code: str, brief: dict | str) -> dict:
    """Validate a brief with the skill's rules; save it as brief.json when valid."""
    code = str(product_code).zfill(7)
    fc_path = _dir(code) / "forecast.json"
    if not fc_path.exists():
        return {"valid": False, "errors": [f"{fc_path} missing — run forecast.write_evidence first"]}
    if isinstance(brief, str):
        try:
            brief = json.loads(brief)
        except json.JSONDecodeError as e:
            return {"valid": False, "errors": [f"brief is not valid JSON: {e}"]}
    brief["product_code"] = code
    fc = style_dna.load_forecast(fc_path)
    errors = style_dna.validate_brief(brief, fc)
    if errors:
        return {"valid": False, "errors": errors}
    out = _dir(code) / "brief.json"
    out.write_text(json.dumps({**brief, "run_id": RUN["run_id"], "validated_at": _now()}, indent=2,
                              ensure_ascii=False), encoding="utf-8")
    return {"valid": True, "errors": [], "path": str(out)}


def do_record_critique(product_code: str, concept_path: str, decision: str, novelty: dict, note: str) -> dict:
    """Append one critic decision for a concept."""
    if decision not in ("approve", "revise"):
        raise ValueError("decision must be 'approve' or 'revise'")
    code = str(product_code).zfill(7)
    rec = {"ts": _now(), "run_id": RUN["run_id"], "product_code": code, "concept_path": concept_path,
           "decision": decision, "note": note, "novelty": novelty}
    with open(_dir(code) / "critic.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"recorded": True, "n_critiques_this_run": sum(
        r["run_id"] == RUN["run_id"] for r in _jsonl(_dir(code) / "critic.jsonl"))}


def do_write_lineage(product_code: str, final_concept_path: str) -> dict:
    """Assemble lineage.json from forecast.json, brief.json, generation_log and critic.jsonl."""
    code = str(product_code).zfill(7)
    d = _dir(code)
    fc = json.loads((d / "forecast.json").read_text(encoding="utf-8"))
    brief_path = d / "brief.json"
    brief = json.loads(brief_path.read_text(encoding="utf-8")) if brief_path.exists() else None
    gens = {Path(r["path"]).resolve(): r for r in generate.read_log(code)}
    final = Path(final_concept_path).resolve()
    if final not in gens:
        raise ValueError(f"{final_concept_path} is not a logged concept of {code}; see image.list_concepts.")
    crits = [r for r in _jsonl(d / "critic.jsonl") if r["run_id"] == RUN["run_id"]]
    attempts = [{"concept": str(p), "prompt": r["prompt"], "seed": r["seed"], "model": r["model"],
                 "run_id": r.get("run_id"), "generated_in_this_run": r.get("run_id") == RUN["run_id"],
                 "is_revision": r.get("is_revision", False),
                 "critique": next(({k: c[k] for k in ("decision", "note", "novelty")} for c in reversed(crits)
                                   if Path(c["concept_path"]).resolve() == p), None)}
                for p, r in gens.items() if r.get("run_id") == RUN["run_id"] or p == final or r.get("note")
                or any(Path(c["concept_path"]).resolve() == p for c in crits)]
    for a in attempts:  # surface manual interventions (e.g. a user-approved extra revision)
        rec = gens[Path(a["concept"])]
        if rec.get("note"):
            a.update(manual=True, forced=rec.get("forced", False), note=rec["note"], input_image=rec["reference_image"])
    fin = gens[final]
    final_crit = next((c for c in reversed(crits) if Path(c["concept_path"]).resolve() == final), None)
    lineage = {
        "run_id": RUN["run_id"], "cutoff": RUN["cutoff"], "product_code": code, "written_at": _now(),
        "forecast": {"path": str(d / "forecast.json"), "rank": fc["rank"],
                     "predicted_units_next_4w": fc["predicted_units_next_4w"], "units_last_4w": fc["units_last_4w"],
                     "shap_drivers": fc["shap_drivers"], "name": fc["attributes"]["prod_name"],
                     "product_type": fc["attributes"]["product_type_name"]},
        "brief": {"path": str(brief_path) if brief else None,
                  "keep": brief["keep"] if brief else None, "change": brief["change"] if brief else None,
                  "rationale": brief["rationale"] if brief else None},
        "final_concept": {"path": str(final), "reference_image": fin["reference_image"], "prompt": fin["prompt"],
                          "prompt_matches_brief": bool(brief) and fin["prompt"] == brief["prompt"],
                          "model": fin["model"], "seed": fin["seed"], "note": fin.get("note"), "generated_in_this_run": fin.get("run_id") == RUN["run_id"],
                          "critic_decision": final_crit["decision"] if final_crit else None,
                          "scores": final_crit["novelty"] if final_crit else None},
        "attempts": attempts,
        "n_new_generations_this_run": sum(a["generated_in_this_run"] for a in attempts),
    }
    out = d / "lineage.json"
    out.write_text(json.dumps(lineage, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"path": str(out), "critic_decision": lineage["final_concept"]["critic_decision"],
            "prompt_matches_brief": lineage["final_concept"]["prompt_matches_brief"]}


@tool("validate_brief", "Validate a style-dna-brief JSON with the skill's rules (skills_lib/style_dna.py). "
      "If valid it is saved as outputs/evidence/<code>/brief.json; otherwise fix the listed errors and call again.",
      {"product_code": str, "brief": dict})
async def validate_brief(args: dict) -> dict:
    """MCP wrapper for do_validate_brief."""
    return _ok(do_validate_brief(args["product_code"], args["brief"]))


@tool("record_critique", "Record the critic's decision for ONE concept: decision 'approve' or 'revise', the "
      "novelty_check result dict, and a one-sentence note (for 'revise': the single revision instruction).",
      {"product_code": str, "concept_path": str, "decision": str, "novelty": dict, "note": str})
async def record_critique(args: dict) -> dict:
    """MCP wrapper for do_record_critique."""
    try:
        return _ok(do_record_critique(args["product_code"], args["concept_path"], args["decision"],
                                      args["novelty"], args["note"]))
    except (ValueError, FileNotFoundError) as e:
        return _err(str(e))


@tool("write_lineage", "Write outputs/evidence/<code>/lineage.json linking forecast.json → brief.json → prompt → "
      "concept → critic scores, for the concept chosen as final. Call once per winner at the end.",
      {"product_code": str, "final_concept_path": str})
async def write_lineage(args: dict) -> dict:
    """MCP wrapper for do_write_lineage."""
    try:
        return _ok(do_write_lineage(args["product_code"], args["final_concept_path"]))
    except (ValueError, FileNotFoundError, KeyError) as e:
        return _err(str(e))


SERVER = create_sdk_mcp_server("run", tools=[validate_brief, record_critique, write_lineage])
TOOLS = ["mcp__run__validate_brief", "mcp__run__record_critique", "mcp__run__write_lineage"]
