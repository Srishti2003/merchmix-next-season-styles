"""Download one catalogue photo per style for the frontend list (outputs/refs/<style_id>/, git-ignored).

  python scripts/fetch_list_photos.py --n 50                    # AW2020 (outputs/predictions.json)
  python scripts/fetch_list_photos.py --n 50 --season SS2020    # outputs/predictions_SS2020.json

Uses the per-image Kaggle download (needs Kaggle credentials). Tries the best-selling colourway first and falls
back to the next colourways when Kaggle has no image for it. Skips styles that already have a photo.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from data_science import select  # noqa: E402


FILES = {"AW2020": "predictions.json", "SS2020": "predictions_SS2020.json"}


def fetch(n: int = 50, season: str = "AW2020") -> int:
    exe = str(Path(sys.executable).with_name("kaggle"))
    styles = sorted(json.loads((config.OUT_DIR / FILES[season]).read_text())["styles"],
                    key=lambda s: s["forecast_rank"])[:n]
    found = 0
    for s in styles:
        dest = config.REFS_DIR / s["style_id"]
        got = next(iter(sorted(dest.glob("*.jpg"))), None)
        for kp in ([] if got else s["images"]["catalogue"]):
            try:
                subprocess.run([exe, "competitions", "download", "-c", select.KAGGLE_COMP, "-f", kp, "-p", str(dest),
                                "-q"], capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired:
                pass
            if (dest / Path(kp).name).exists():
                got = dest / Path(kp).name
                break
        found += got is not None
        print(f"{s['style_id']} {s['name']}: {got.name if got else 'no photo on Kaggle'}", flush=True)
    print(f"{found}/{len(styles)} styles have a photo")
    return found


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--season", choices=list(FILES), default="AW2020")
    args = ap.parse_args()
    fetch(args.n, args.season)
