"""Check that every relative link and image in the docs resolves (files, folders and #anchors).

  python scripts/check_links.py              # README.md, WRITEUP.md, API.md; exit code 1 on a broken link
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "WRITEUP.md", "API.md"]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMG = re.compile(r"<img\s[^>]*src=\"([^\"]+)\"")


def slug(heading: str) -> str:
    """GitHub-style anchor for a markdown heading."""
    s = heading.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)  # drop punctuation (keeps letters, digits, _, -, spaces)
    return s.replace(" ", "-")


def anchors(md: Path) -> set[str]:
    text = re.sub(r"```.*?```", "", md.read_text(encoding="utf-8"), flags=re.S)
    return {slug(m[1]) for m in re.finditer(r"^#{1,6}\s+(.+)$", text, flags=re.M)}


def check(doc: Path) -> list[str]:
    text = re.sub(r"```.*?```", "", doc.read_text(encoding="utf-8"), flags=re.S)  # skip code blocks
    bad = []
    for target in [*LINK.findall(text), *IMG.findall(text)]:
        if re.match(r"^(https?:|mailto:)", target):
            continue
        path, _, frag = target.partition("#")
        dest = (doc.parent / path).resolve() if path else doc
        if not dest.exists():
            bad.append(f"{doc.name}: missing {target}")
        elif frag and dest.suffix == ".md" and frag not in anchors(dest):
            bad.append(f"{doc.name}: no heading for #{frag} in {dest.name}")
    return bad


def main() -> int:
    bad, n = [], 0
    for name in DOCS:
        doc = ROOT / name
        text = re.sub(r"```.*?```", "", doc.read_text(encoding="utf-8"), flags=re.S)
        n += len(LINK.findall(text)) + len(IMG.findall(text))
        bad += check(doc)
    print(f"checked {n} links/images in {', '.join(DOCS)}: {len(bad)} broken")
    for b in bad:
        print("  " + b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
