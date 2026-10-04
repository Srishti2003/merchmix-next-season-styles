"""Render WRITEUP.md to WRITEUP.pdf (images and the Mermaid diagram included) with headless Chromium.

  pip install markdown playwright && python -m playwright install chromium   # dev-only tools
  python scripts/build_writeup_pdf.py
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: "Segoe UI", "DejaVu Sans", Arial, sans-serif; font-size: 10pt; line-height: 1.42; color: #1b1b1b; }
h1 { font-size: 18pt; margin: 0 0 6pt; } h2 { font-size: 13pt; margin: 14pt 0 4pt; border-bottom: 1px solid #ddd; padding-bottom: 2pt; }
p, li { margin: 3pt 0; } ul { padding-left: 16pt; margin: 3pt 0; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.6pt; background: #f3f3f1; padding: 0 2pt; border-radius: 2pt; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 8.8pt; width: 100%; page-break-inside: avoid; }
th, td { border: 1px solid #d6d6d2; padding: 3pt 5pt; text-align: left; vertical-align: top; } th { background: #f3f3f1; }
img { max-width: 100%; display: block; margin: 6pt auto; page-break-inside: avoid; }
.mermaid { text-align: center; margin: 6pt 0; page-break-inside: avoid; }
a { color: #1d5fb4; text-decoration: none; }
"""


def normalise_lists(md_text: str) -> str:
    """Python-Markdown needs 4-space nested lists and a blank line before a list; GitHub doesn't."""
    out, prev = [], ""
    for line in md_text.splitlines():
        m = re.match(r"^( +)([-*] |\d+\. )", line)
        if m:
            line = " " * (len(m[1]) * 2) + line.lstrip()
        if re.match(r"^([-*] |\d+\. )", line) and prev.strip() and not re.match(r"^\s*([-*] |\d+\. )", prev) \
                and not prev.startswith("  "):
            out.append("")
        out.append(line)
        prev = line
    return "\n".join(out) + "\n"


def to_html(md_text: str) -> str:
    md_text = normalise_lists(md_text)

    def mermaid(m: re.Match) -> str:
        return f'<div class="mermaid">{html.escape(m[1])}</div>'
    md_text = re.sub(r"```mermaid\n(.*?)```", mermaid, md_text, flags=re.S)
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    return f"""<!doctype html><html><head><meta charset="utf-8"><base href="{ROOT.as_uri()}/">
<style>{CSS}</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad: true, theme: "neutral", flowchart: {{useMaxWidth: true}}}});</script>
</head><body>{body}</body></html>"""


def main() -> None:
    page_html = to_html((ROOT / "WRITEUP.md").read_text(encoding="utf-8"))
    tmp = ROOT / "WRITEUP.render.html"
    tmp.write_text(page_html, encoding="utf-8")
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page()
            pg.goto(tmp.as_uri(), wait_until="networkidle")
            pg.wait_for_function("document.querySelectorAll('.mermaid svg').length === document.querySelectorAll('.mermaid').length",
                                 timeout=30000)
            pg.wait_for_function("Array.from(document.images).every(i => i.complete && i.naturalWidth > 0)", timeout=30000)
            pg.pdf(path=str(ROOT / "WRITEUP.pdf"), format="A4", print_background=True, prefer_css_page_size=True)
            b.close()
    finally:
        tmp.unlink(missing_ok=True)
    print("wrote WRITEUP.pdf")


if __name__ == "__main__":
    main()
