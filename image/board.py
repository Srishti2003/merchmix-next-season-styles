"""Compose the final board: one column per winner, reference → concept, with captions."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

import config

W = 1920
MARGIN, GUTTER = 48, 36
COL_W = (W - 2 * MARGIN - 2 * GUTTER) // 3
IMG_W = (COL_W - 44) // 2
IMG_H = int(IMG_W * 1.5)
BG, INK, INK2, RULE, ACCENT, CARD = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#ffffff"
OK_COL, BAD_COL = "#1baf7a", "#e34948"
WHY_LINES, CHANGED_LINES = 5, 7


def safe_save(img: Image.Image, path: Path, attempts: int = 6) -> Path:
    """Save a PNG, surviving a viewer (e.g. Windows Photos) briefly locking the file: write to a temp file and
    atomically replace, retrying a few times."""
    import os
    import time

    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.png")
    img.save(tmp)
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return path
        except OSError:
            time.sleep(0.5 * (i + 1))
    raise OSError(f"Could not replace {path} (open in another program?). New image left at {tmp}.")


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Segoe UI / Arial if available, else PIL's default."""
    for name in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _wrap(text: str, font: ImageFont.FreeTypeFont, width: int, max_lines: int) -> list[str]:
    """Greedy word-wrap to a pixel width; ellipsis on overflow."""
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if font.getlength(trial) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while font.getlength(lines[-1] + " …") > width and " " in lines[-1]:
            lines[-1] = lines[-1].rsplit(" ", 1)[0]
        lines[-1] += " …"
    return lines


def _fit(path: str | Path, w: int, h: int) -> Image.Image:
    """Letterbox an image into w×h on white (product shots keep their full silhouette)."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), CARD)
    canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
    return canvas


def compose_board(entries: list[dict], out_path: Path | None = None,
                  title: str = "Next-Season Concepts",
                  subtitle: str = "Top-3 forecast winners (next 4 weeks after 22 Sep 2020) → AI-generated next-season concepts") -> Path:
    """Render up to 3 entries into one PNG.

    Each entry: {product_code, ref_path, concept_path, why_it_won, what_changed} plus optional
    ``label`` (e.g. "#1 Pluto RW slacks · Trousers") and ``status`` / ``status_ok`` (a small tag under the
    concept, e.g. "Critic: approved", green if status_ok else red). Returns the saved path (default outputs/final_board.png;
    outputs/mock_run/final_board.png when HM_EVIDENCE_DIR points at the mock run).
    """
    if not 1 <= len(entries) <= 3:
        raise ValueError(f"compose_board takes 1–3 entries, got {len(entries)}.")
    for e in entries:
        for k in ("product_code", "ref_path", "concept_path", "why_it_won", "what_changed"):
            if not e.get(k):
                raise ValueError(f"Entry {e.get('product_code', '?')} is missing '{k}'.")
        for k in ("ref_path", "concept_path"):
            if not Path(e[k]).exists():
                raise FileNotFoundError(f"{k} not found: {e[k]}")
    out_path = Path(out_path or config.EVIDENCE_DIR.parent / "final_board.png")  # outputs/ (or mock_run/)

    f_title, f_sub = _font(56, True), _font(26)
    f_head, f_code, f_lab = _font(30, True), _font(22), _font(20, True)
    f_cap_h, f_cap = _font(21, True), _font(21)
    top = MARGIN + 150
    img_y = top + 96
    has_status = any(e.get("status") for e in entries)
    cap_y = img_y + IMG_H + (92 if has_status else 52)
    H = cap_y + 2 * (30 + 14) + (WHY_LINES + CHANGED_LINES) * 27 + MARGIN

    board = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(board)
    d.text((MARGIN, MARGIN), title, font=f_title, fill=INK)
    d.text((MARGIN, MARGIN + 76), subtitle, font=f_sub, fill=INK2)
    d.line([(MARGIN, top - 22), (W - MARGIN, top - 22)], fill=RULE, width=2)

    for i, e in enumerate(entries):
        x = MARGIN + i * (COL_W + GUTTER)
        label = e.get("label") or f"#{i + 1}"
        d.text((x, top), _wrap(label, f_head, COL_W, 1)[0], font=f_head, fill=INK)
        d.text((x, top + 44), f"style {str(e['product_code']).zfill(7)}", font=f_code, fill=INK2)
        board.paste(_fit(e["ref_path"], IMG_W, IMG_H), (x, img_y))
        cx = x + COL_W - IMG_W
        board.paste(_fit(e["concept_path"], IMG_W, IMG_H), (cx, img_y))
        d.rectangle([x, img_y, x + IMG_W - 1, img_y + IMG_H - 1], outline=RULE, width=2)
        d.rectangle([cx, img_y, cx + IMG_W - 1, img_y + IMG_H - 1], outline=ACCENT, width=3)
        ay = img_y + IMG_H // 2
        ax0, ax1 = x + IMG_W + 8, cx - 8
        d.line([(ax0, ay), (ax1 - 4, ay)], fill=INK2, width=3)
        d.polygon([(ax1, ay), (ax1 - 12, ay - 8), (ax1 - 12, ay + 8)], fill=INK2)
        d.text((x, img_y + IMG_H + 12), "WINNING STYLE", font=f_lab, fill=INK2)
        d.text((cx, img_y + IMG_H + 12), "NEW CONCEPT", font=f_lab, fill=ACCENT)
        if e.get("status"):
            f_tag = _font(18, True)
            tw = int(f_tag.getlength(e["status"]))
            ty = img_y + IMG_H + 44
            d.rounded_rectangle([cx, ty, cx + tw + 20, ty + 30], radius=6,
                                fill=OK_COL if e.get("status_ok") else BAD_COL)
            d.text((cx + 10, ty + 4), e["status"], font=f_tag, fill="#ffffff")

        y = cap_y
        for head, body, n in (("Why it won", e["why_it_won"], WHY_LINES),
                              ("What changed", e["what_changed"], CHANGED_LINES)):
            d.text((x, y), head, font=f_cap_h, fill=INK)
            y += 30
            for line in _wrap(body, f_cap, COL_W, n):
                d.text((x, y), line, font=f_cap, fill=INK2)
                y += 27
            y += 14

    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe_save(board, out_path)
    return out_path
