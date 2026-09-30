"""Central configuration: paths, dates, horizon, seeds and API keys (loaded from .env)."""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

ROOT: Path = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# --- Data ---------------------------------------------------------------
DATA_DIR: Path = Path(os.getenv("HM_DATA_DIR") or ROOT / "data")  # scripts/convert_data.py writes here
TX_GLOB: str = (DATA_DIR / "tx" / "part_*.parquet").as_posix()  # DuckDB-friendly glob
ARTICLES_PATH: str = (DATA_DIR / "articles.parquet").as_posix()

# --- Outputs ------------------------------------------------------------
OUT_DIR: Path = ROOT / "outputs"
CACHE_DIR: Path = OUT_DIR / "cache"
EVIDENCE_DIR: Path = Path(os.getenv("HM_EVIDENCE_DIR") or OUT_DIR / "evidence")  # override for mock/test runs
REFS_DIR: Path = OUT_DIR / "refs"
FIGURES_DIR: Path = OUT_DIR / "figures"
MODELS_DIR: Path = OUT_DIR / "models"

# --- Forecast setup -----------------------------------------------------
# Convention: weeks run Wednesday -> Tuesday. Every cutoff is a Wednesday and
# EXCLUSIVE: features use t_dat < cutoff; target = [cutoff, cutoff + HORIZON_WEEKS).
HORIZON_WEEKS: int = 4
FIRST_WEEK: date = date(2018, 9, 26)    # first full Wed->Tue week in the data
FINAL_CUTOFF: date = date(2020, 9, 23)  # day after last transaction (2020-09-22)
VALID_CUTOFF: date = date(2020, 8, 26)  # target = last 4 full weeks of data
LY_OFFSET_DAYS: int = 364               # "last year" = 52 weeks back, same weekday
SEED: int = 42

# --- Secrets ------------------------------------------------------------
def _secret(name: str) -> str | None:
    """Return a real-looking secret, or None. Placeholders ('sk-ant-...', empty) are also removed from
    os.environ so child processes (e.g. the Claude Agent SDK) never try them — without a real
    ANTHROPIC_API_KEY the SDK falls back to the local Claude Code login."""
    v = (os.getenv(name) or "").strip()
    if not v or "..." in v or len(v) < 20:
        os.environ.pop(name, None)
        return None
    return v


ANTHROPIC_API_KEY: str | None = _secret("ANTHROPIC_API_KEY")
REPLICATE_API_TOKEN: str | None = _secret("REPLICATE_API_TOKEN")
HF_TOKEN: str | None = _secret("HF_TOKEN")


def assert_cutoff(d: date) -> date:
    """Raise ValueError unless ``d`` is a Wednesday (valid exclusive cutoff); return ``d``."""
    if d.weekday() != 2:
        raise ValueError(f"Cutoff {d} is a {d:%A}; cutoffs must be Wednesdays (weeks run Wed->Tue).")
    return d


def last_year(d: date) -> date:
    """Return the same weekday 52 weeks earlier (``d - 364 days``)."""
    return d - timedelta(days=LY_OFFSET_DAYS)


for _d in (FIRST_WEEK, FINAL_CUTOFF, VALID_CUTOFF):
    assert_cutoff(_d)


def ensure_dirs() -> None:
    """Create all output directories if they do not exist."""
    for d in (OUT_DIR, CACHE_DIR, EVIDENCE_DIR, REFS_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
