"""Evidence sheet: one row per winner — sales curve · model drivers · reference → concept · KEEP/CHANGE.

  python scripts/evidence_sheet.py --run 20260928-005623

Reads only existing files (lineage.json, forecast.json, brief.json, sales_curve.png, captions.json);
no model runs, no image generation. Writes outputs/evidence_sheet.png (1920 px wide).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402

import config  # noqa: E402
from data_science import data, select  # noqa: E402
from image.board import _fit, _font, _wrap, safe_save  # noqa: E402
from scripts.finalize_run import display_name  # noqa: E402

W, M = 1920, 40
ROW_H = 430
FOOT_H, PAD = 120, 36
BG, INK, INK2, RULE, OK_COL, BAD_COL, ACCENT = ("#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#1baf7a", "#e34948",
                                                "#2a78d6")
CODES = ["0751471", "0762846", "0685814"]


SHAP_NOTE = ("Model predicts a multiplier on last-week × 4; a big recent week pulls it down (mean reversion). "
             "Top 5 factors shown; others + base rate make up the rest.")


def text_block(d: ImageDraw.ImageDraw, x: int, y: int, w: int, head: str, lines: list[str], size: int = 17,
               max_lines: int = 14, note: str | None = None) -> int:
    """Heading (+ optional small grey note) + wrapped bullet lines; returns the y after the block."""
    d.text((x, y), head, font=_font(18, True), fill=INK)
    y += 26
    if note:
        for part in _wrap(note, _font(13), w, 3):
            d.text((x, y), part, font=_font(13), fill="#8a8984")
            y += 17
        y += 6
    f, used = _font(size), 0
    for line in lines:
        for i, part in enumerate(_wrap(line, f, w - 14, 4)):
            if used >= max_lines:
                return y
            d.text((x + (14 if i else 0), y), ("• " if i == 0 else "") + part, font=f, fill=INK2)
            y += size + 6
            used += 1
    return y + 8


def row(code: str, run_id: str, captions: dict) -> Image.Image:
    """Render one winner's evidence row."""
    d_ = config.EVIDENCE_DIR / code
    lin = json.loads((d_ / "lineage.json").read_text(encoding="utf-8"))
    fc = json.loads((d_ / "forecast.json").read_text(encoding="utf-8"))
    brief = json.loads((d_ / "brief.json").read_text(encoding="utf-8"))
    cap = captions[code]
    img = Image.new("RGB", (W, ROW_H), BG)
    d = ImageDraw.Draw(img)
    approved = lin["final_concept"]["critic_decision"] == "approve"
    scores = lin["final_concept"]["scores"] or {}
    d.text((M, 12), f"#{fc['rank']} {display_name(fc['attributes']['prod_name'])} · "
                    f"{fc['attributes']['product_type_name']} · style {code}", font=_font(24, True), fill=INK)

    # 1) sales curve (existing PNG from data_science.select)
    y0 = 52
    img.paste(_fit(d_ / "sales_curve.png", 520, 250), (M, y0))
    d.text((M, y0 + 258), f"Forecast {fc['predicted_units_next_4w']:,.0f} units, 23 Sep–20 Oct 2020",
           font=_font(17, True), fill=INK)
    d.text((M, y0 + 282), f"last 4 weeks {fc['units_last_4w']:,} · last-week × 4 run-rate "
                          f"{fc['naive_run_rate_units']:,} ({fc['predicted_units_next_4w'] / fc['naive_run_rate_units'] - 1:+.0%})",
           font=_font(16), fill=INK2)
    d.text((M, y0 + 304), f"sells in {fc['attributes']['n_colours']} colours", font=_font(16), fill=INK2)

    # 2) model drivers
    x2 = M + 540
    text_block(d, x2, y0, 360, "Top SHAP drivers (× on the run-rate)", fc["shap_drivers"][:5], size=15,
               note=SHAP_NOTE)

    # 3) reference -> concept
    x3 = x2 + 380
    iw, ih = 190, 285
    img.paste(_fit(select.reference_images(code)[0], iw, ih), (x3, y0))
    img.paste(_fit(lin["final_concept"]["path"], iw, ih), (x3 + iw + 16, y0))
    d.rectangle([x3 + iw + 16, y0, x3 + 2 * iw + 15, y0 + ih - 1], outline=ACCENT, width=3)
    d.text((x3, y0 + ih + 6), "reference", font=_font(15), fill=INK2)
    d.text((x3 + iw + 16, y0 + ih + 6), Path(lin["final_concept"]["path"]).name, font=_font(15), fill=ACCENT)
    tag = f"Critic: {'approved' if approved else 'not approved'} · CLIP {scores.get('max_sim_to_refs', float('nan')):.3f}"
    tw = int(_font(15, True).getlength(tag))
    d.rounded_rectangle([x3, y0 + ih + 30, x3 + tw + 16, y0 + ih + 54], radius=5, fill=OK_COL if approved else BAD_COL)
    d.text((x3 + 8, y0 + ih + 33), tag, font=_font(15, True), fill="#ffffff")

    # 4) KEEP / CHANGE (visible changes only; not-rendered brief items listed separately)
    x4 = x3 + 2 * iw + 40
    wk = W - x4 - M
    y = text_block(d, x4, y0, wk, "KEEP (as written in brief.json)", [k["trait"] for k in brief["keep"]], max_lines=6)
    y = text_block(d, x4, y, wk, "CHANGE (visible in the concept)", [cap["what_changed"]], max_lines=3)
    if cap["not_visible"]:
        text_block(d, x4, y, wk, "Briefed but not rendered", cap["not_visible"], size=15, max_lines=4)
    d.line([(M, ROW_H - 2), (W - M, ROW_H - 2)], fill=RULE, width=2)
    return img


def insight_line(codes: list[str]) -> str:
    """One-line pattern across the winners, built from forecast.json + the weekly sales table (checked, not assumed)."""
    rise, vs_l4, disc = [], [], []
    for c in codes:
        fc = json.loads((config.EVIDENCE_DIR / c / "forecast.json").read_text(encoding="utf-8"))
        u = data.sales_curve(c, weeks=8)["units"].tolist()
        rise.append(sum(u[4:]) / max(1, sum(u[:4])))
        vs_l4.append(fc["growth_vs_last_4w"])
        disc += [int(m.group(1)) for d in fc["shap_drivers"] if (m := re.match(r"discount .*? = (\d+)%", d))]
    all_rose, all_below, near_full = min(rise) > 1.5, max(vs_l4) < 0, bool(disc) and max(disc) <= 5
    if not (all_rose and all_below and near_full):
        return "Winners differ in momentum and price position; see each row."
    return (f"All three are early-autumn risers (last 4 weeks = {min(rise):.1f}–{max(rise):.1f}× the 4 weeks before) whose "
            f"September rise the model partly discounts (forecast {', '.join(f'{g:+.0%}' for g in vs_l4)} vs last 4 weeks); "
            f"they win on sustained near-full-price volume ({min(disc)}–{max(disc)}% below peak price), not markdowns.")


def footer_lines() -> list[str]:
    """Backtest / movers summary parsed from outputs/figures/eval_table.md and movers.md (exact numbers)."""
    ev = (config.FIGURES_DIR / "eval_table.md").read_text(encoding="utf-8")
    mv = (config.FIGURES_DIR / "movers.md").read_text(encoding="utf-8")
    bt = dict(re.findall(r"^\| (LightGBM regressor|Last week × 4|Last 4 weeks units) \| ([\d.]+) ± [\d.]+ \|", ev, re.M))
    val = dict(re.findall(r"^\| (LightGBM regressor|Last 4 weeks units) \|(?: [^|]+ \|){3} \**([\d.]+)\** \|", ev, re.M))
    wins = dict(re.findall(r"^- vs (Last week × 4|Last 4 weeks units) — ndcg@50: model wins \*\*(\d+/\d+)\*\*", ev, re.M))
    n_cut = re.search(r"Rolling backtest: (\d+) weekly cutoffs", ev).group(1)
    promo = re.search(r"\(promoted\): the model was closer to the actual rank for (\d+/\d+)", mv).group(1)
    demo = re.search(r"\(demoted\): the model was closer to the actual rank for (\d+/\d+)", mv).group(1)
    m, lw, l4 = bt["LightGBM regressor"], bt["Last week × 4"], bt["Last 4 weeks units"]
    return [
        f"Backtest ({n_cut} rolling weekly cutoffs, NDCG@50 mean): model {m} vs last-week × 4 {lw} ({wins['Last week × 4']} "
        f"cutoff wins) and vs last-4-weeks {l4} ({wins['Last 4 weeks units']} wins). Last-4-weeks is beaten clearly only "
        f"on the validation week ({val['LightGBM regressor']} vs {val['Last 4 weeks units']}).",
        f"Gains come mainly from demoting fading styles: where the model disagrees with last-week × 4, its demotions were "
        f"right {demo}, its promotions {promo} (outputs/figures/movers.md).",
    ]


def main() -> None:
    """Compose the three rows under a title and save outputs/evidence_sheet.png."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    captions = json.loads((config.OUT_DIR / "runs" / a.run / "captions.json").read_text(encoding="utf-8"))
    rows = [row(c, a.run, captions) for c in CODES]
    head = 168
    sheet = Image.new("RGB", (W, head + ROW_H * len(rows) + FOOT_H + PAD), BG)
    d = ImageDraw.Draw(sheet)
    d.text((M, 24), "Evidence: from forecast to concept (Autumn 2020)", font=_font(40, True), fill=INK)
    d.text((M, 74), f"Run {a.run} · every number from outputs/evidence/<code>/forecast.json, lineage.json and "
                    "brief.json · no stock data (sold-in-last-2-weeks is the availability proxy)",
           font=_font(18), fill=INK2)
    for j, line in enumerate(_wrap(insight_line(CODES), _font(18, True), W - 2 * M, 2)):
        d.text((M, 104 + j * 26), line, font=_font(18, True), fill=INK)
    for i, r in enumerate(rows):
        sheet.paste(r, (0, head + i * ROW_H))
    fy = head + ROW_H * len(rows) + 12
    d.rounded_rectangle([M, fy, W - M, fy + FOOT_H - 12], radius=8, fill="#f0efec")
    d.text((M + 18, fy + 12), "Model vs baselines", font=_font(18, True), fill=INK)
    for j, line in enumerate(footer_lines()):
        d.text((M + 18, fy + 42 + j * 28), line, font=_font(16), fill=INK2)
    out = config.OUT_DIR / "evidence_sheet.png"
    safe_save(sheet, out)
    print(out)


if __name__ == "__main__":
    main()
