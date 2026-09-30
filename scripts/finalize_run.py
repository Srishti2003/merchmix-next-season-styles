"""Finish a reviewed run: set the final concept per winner, rewrite lineage.json, review sheet, final board.

  python scripts/finalize_run.py --run 20260928-005623 --final 0751471=concept_2.png 0762846=concept_1.png \
      0685814=concept_2.png

Everything is read from the evidence files (forecast.json, brief.json, generation_log.jsonl, critic.jsonl),
so the board captions carry the same numbers as the lineage.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageOps  # noqa: E402

import config  # noqa: E402
from agents import runtools  # noqa: E402
from forecasting import select  # noqa: E402
from image.board import _font, _wrap, compose_board  # noqa: E402

GREEN, RED, GREY, INK = "#1baf7a", "#e34948", "#52514e", "#0b0b0b"


def critiques(code: str, run_id: str) -> list[dict]:
    """Critic records for one style in one run, oldest first."""
    f = config.EVIDENCE_DIR / code / "critic.jsonl"
    return [r for r in (json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip())
            if r["run_id"] == run_id]


def review_sheet(run_id: str, lineages: list[dict], out: Path) -> Path:
    """Reference + every concept judged in the run, with decision, CLIP score and a FINAL tag."""
    W, H, rows = 300, 450, []
    for lin in lineages:
        code = lin["product_code"]
        ref = str(select.reference_images(code)[0])
        items = [(ref, "reference", None, False)]
        for c in critiques(code, run_id):
            final = Path(c["concept_path"]).resolve() == Path(lin["final_concept"]["path"]).resolve()
            items.append((c["concept_path"], f"{Path(c['concept_path']).name}: {c['decision'].upper()} · CLIP "
                          f"{c['novelty']['max_sim_to_refs']:.3f} ({c['novelty']['verdict']})", c, final))
        row = Image.new("RGB", (20 + len(items) * (W + 20), H + 120), "#fcfcfb")
        d = ImageDraw.Draw(row)
        f = lin["forecast"]
        d.text((20, 8), f"{code} {display_name(f['name'])} — #{f['rank']}, {f['predicted_units_next_4w']:,.0f} units forecast",
               font=_font(22, True), fill=INK)
        for i, (path, cap, c, final) in enumerate(items):
            x = 20 + i * (W + 20)
            row.paste(ImageOps.contain(Image.open(path).convert("RGB"), (W, H)), (x, 44))
            color = GREY if c is None else GREEN if c["decision"] == "approve" else RED
            y = 44 + H + 6
            for line in _wrap(cap, _font(15), W, 2):
                d.text((x, y), line, font=_font(15), fill=color)
                y += 19
            if final:
                d.rectangle([x, y + 4, x + 64, y + 26], fill=INK)
                d.text((x + 8, y + 5), "FINAL", font=_font(15, True), fill="#ffffff")
        rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "#fcfcfb")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(out)
    return out


def display_name(prod_name: str) -> str:
    """Product name for display: drop H&M's variant suffixes like ' (1)'."""
    return re.sub(r"\s*\(\d+\)\s*$", "", prod_name).strip()


def top_upward_driver(drivers: list[str]) -> str | None:
    """Plain-English version of the strongest SHAP driver that RAISES the forecast (None if none do)."""
    for d in drivers:
        m = re.match(r"(.+?) = (.+?) → raises the forecast ×([\d.]+)", d)
        if not m:
            continue
        label, value, mult = m.groups()
        if label.startswith("discount"):
            return f"still near full price, only {value} below its peak price (×{mult})"
        return f"{label} = {value} (×{mult})"
    return None


def board_entry(lin: dict, n_scored: int, cap: dict) -> dict:
    """Board column: 'why it won' from forecast.json (volume + model reasons); 'what changed' from the
    visually checked caption (captions.json) — never from brief items the image does not show."""
    code = lin["product_code"]
    fc = json.loads(Path(lin["forecast"]["path"]).read_text(encoding="utf-8"))
    approved = lin["final_concept"]["critic_decision"] == "approve"
    pred, run_rate = fc["predicted_units_next_4w"], fc["naive_run_rate_units"]
    driver = top_upward_driver(fc["shap_drivers"])
    why = (f"Rank #{fc['rank']} of {n_scored:,} styles scored (one per garment group): {pred:,.0f} units forecast "
           f"for 23 Sep–20 Oct 2020 ({fc['units_last_4w']:,} in the last 4 weeks). Model: {pred / run_rate - 1:+.0%} "
           f"vs the last-week × 4 run-rate of {run_rate:,}" + (f"; top upward driver: {driver}." if driver else "."))
    what = f"{cap['what_changed']} Kept: {cap['kept']}."
    if cap.get("extra_note"):
        what += f" {cap['extra_note']}"
    return {"product_code": code, "ref_path": str(select.reference_images(code)[0]),
            "concept_path": lin["final_concept"]["path"], "why_it_won": why, "what_changed": what,
            "label": f"#{fc['rank']} {display_name(fc['attributes']['prod_name'])} · "
                     f"{fc['attributes']['product_type_name']}",
            "status": "Critic: approved" if approved else "Critic: not approved", "status_ok": approved}


def main() -> None:
    """Rewrite lineage for the chosen finals, then the review sheet and the final board."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--final", nargs="+", required=True, help="CODE=concept_N.png, in rank order")
    a = ap.parse_args()
    run_dir = config.OUT_DIR / "runs" / a.run
    meta = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    runtools.RUN.update(run_id=a.run, cutoff=meta["cutoff"])
    lineages = []
    for item in a.final:
        code, name = item.split("=")
        res = runtools.do_write_lineage(code, str(config.EVIDENCE_DIR / code / name))
        lineages.append(json.loads(Path(res["path"]).read_text(encoding="utf-8")))
        print(f"{code}: final {name} | critic {res['critic_decision']} | prompt_matches_brief "
              f"{res['prompt_matches_brief']}")
    print("review sheet ->", review_sheet(a.run, lineages, run_dir / "review_concepts.png"))
    n_scored = len(pd.read_parquet(config.CACHE_DIR / "preds_20200923.parquet"))
    captions = json.loads((run_dir / "captions.json").read_text(encoding="utf-8"))
    entries = []
    for lin in lineages:
        cap = captions[lin["product_code"]]
        if cap["concept"] != Path(lin["final_concept"]["path"]).name:
            raise ValueError(f"captions.json describes {cap['concept']} but the final concept of "
                             f"{lin['product_code']} is {Path(lin['final_concept']['path']).name} — re-check by eye.")
        entry = board_entry(lin, n_scored, cap)
        entries.append(entry)
        lin["board_caption"] = {"why_it_won": entry["why_it_won"], "what_changed": entry["what_changed"],
                                "status": entry["status"], "checked_against_image": True,
                                "brief_changes_not_visible": cap["not_visible"]}
        path = config.EVIDENCE_DIR / lin["product_code"] / "lineage.json"
        path.write_text(json.dumps(lin, indent=2, ensure_ascii=False), encoding="utf-8")
    board = compose_board(entries, title="Next-Season Concepts: Autumn 2020",
                          subtitle="Top-3 forecast winners for 23 Sep–20 Oct 2020 (4-week horizon) → AI-generated "
                                   "next-season concepts")
    print("board ->", board)


if __name__ == "__main__":
    main()
