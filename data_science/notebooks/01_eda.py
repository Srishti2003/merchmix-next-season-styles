# %% [markdown]
# # 01 — EDA: seasonality, category mix, style lifetime, new-style share
# Run from the repo root: `python data_science/notebooks/01_eda.py`. Figures go to outputs/figures/.

# %%
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from data_science import data  # noqa: E402

FIG = config.FIGURES_DIR
FIG.mkdir(parents=True, exist_ok=True)

# Reference palette (light mode): categorical slots in fixed order, ink tokens, surface.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, BAND, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#f0efec", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def kfmt(ax: plt.Axes, axis: str = "y") -> None:
    """Format an axis in thousands (e.g. 250k)."""
    f = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:,.0f}k")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


# %% Load
sw = data.style_weekly()
attrs = data.style_attributes()
sw = sw.merge(attrs[["product_code", "index_group_name", "prod_name", "product_type_name"]],
              on="product_code", how="left")
sw["season"] = [data.season_label((w + timedelta(days=3)).date()) for w in sw["week_start"]]
weeks = sw.groupby("week_start", as_index=False).agg(units=("units", "sum"))
weeks["season"] = [data.season_label((w + timedelta(days=3)).date()) for w in weeks["week_start"]]
print(f"{len(sw):,} style-weeks | {sw.product_code.nunique():,} styles | {len(weeks)} weeks")

# %% Fig 1 — total weekly units with season bands
fig, ax = plt.subplots(figsize=(11, 4))
runs = weeks.groupby((weeks.season != weeks.season.shift()).cumsum())
for i, (_, r) in enumerate(runs):
    x0, x1 = r.week_start.min(), r.week_start.max() + timedelta(days=7)
    if r.season.iloc[0].startswith(("winter", "summer")):
        ax.axvspan(x0, x1, color=BAND, lw=0)
    short = r.season.iloc[0].replace("autumn", "Aut").replace("winter", "Win") \
        .replace("spring", "Spr").replace("summer", "Sum")
    ax.text(x0 + (x1 - x0) / 2, 1.01, short, transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8, color=INK2)
ax.plot(weeks.week_start, weeks.units, color=SERIES[0], lw=2)
pk = weeks.loc[weeks.units.idxmax()]
ax.plot(pk.week_start, pk.units, "o", ms=8, color=SERIES[0], mec=SURFACE, mew=2)
ax.annotate(f"peak {pk.units / 1e3:,.0f}k\nweek of {pk.week_start:%d %b %Y}", (pk.week_start, pk.units),
            xytext=(10, -5), textcoords="offset points", fontsize=8, color=INK2, va="top")
kfmt(ax)
ax.set_ylim(0, None)
ax.set_ylabel("units / week")
ax.set_title("Total weekly units sold (Wed→Tue weeks; shaded = winter & summer)", pad=18)
fig.savefig(FIG / "01_weekly_units.png")
plt.close(fig)

# %% Fig 2 — avg weekly units by index group per season (seasons with >= 4 weeks)
season_weeks = weeks.groupby("season").size()
full_seasons = [s for s in weeks.season.unique() if season_weeks[s] >= 4]
ig = (sw[sw.season.isin(full_seasons)]
      .groupby(["season", "index_group_name"], observed=True).units.sum().unstack())
ig = ig.div(season_weeks[ig.index], axis=0).loc[full_seasons]
groups = ig.sum().sort_values(ascending=False).index.tolist()
fig, ax = plt.subplots(figsize=(11, 4.2))
w = 0.8 / len(groups)
x = np.arange(len(full_seasons))
for j, g in enumerate(groups):
    ax.bar(x + (j - (len(groups) - 1) / 2) * w, ig[g], width=w - 0.02, color=SERIES[j],
           label=g, edgecolor=SURFACE, linewidth=1)
ax.set_xticks(x, full_seasons, fontsize=9)
ax.grid(axis="x", visible=False)
kfmt(ax)
ax.set_ylabel("avg units / week")
ax.legend(ncol=len(groups), loc="upper left", bbox_to_anchor=(0, 1.0), fontsize=9)
ax.set_ylim(0, ig.values.max() * 1.15)
ax.set_title("Average weekly units by index group and season")
fig.savefig(FIG / "02_index_group_by_season.png")
plt.close(fig)

# %% Fig 3 — style lifetime (first→last selling week), styles launched after an 8-week burn-in
life = sw.groupby("product_code").agg(first=("week_start", "min"), last=("week_start", "max"),
                                      active=("week_start", "size"))
life["span"] = (life["last"] - life["first"]).dt.days // 7 + 1
burn_in = pd.Timestamp(config.FIRST_WEEK + timedelta(weeks=8))
last_week = pd.Timestamp(data.LAST_WEEK)
lf = life[life["first"] >= burn_in]
still_selling = (lf["last"] >= last_week - timedelta(weeks=3)).mean()
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(lf.span, bins=np.arange(1, lf.span.max() + 2, 2), color=SERIES[0], edgecolor=SURFACE, lw=1)
med = lf.span.median()
ax.axvline(med, color=INK2, lw=1, ls="--")
ax.text(med + 1, ax.get_ylim()[1] * 0.92, f"median {med:.0f} weeks", color=INK2, fontsize=9)
ax.set_xlabel("weeks from first to last sale")
ax.set_ylabel("styles")
ax.set_title(f"Style selling span (styles launched ≥ {burn_in:%d %b %Y}; "
             f"{still_selling:.0%} still selling at data end)", fontsize=11)
fig.savefig(FIG / "03_style_lifetime.png")
plt.close(fig)

# %% Fig 4 — top-20 styles in the last 4 weeks
l4_start = pd.Timestamp(config.FINAL_CUTOFF - timedelta(weeks=4))
l4 = (sw[sw.week_start >= l4_start].groupby("product_code")
      .agg(units=("units", "sum"), prod_name=("prod_name", "first"),
           ptype=("product_type_name", "first")).nlargest(20, "units").iloc[::-1])
labels = [f"{n[:24]} · {t} ({c})" for c, n, t in zip(l4.index, l4.prod_name, l4.ptype)]
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.barh(labels, l4.units, color=SERIES[0], height=0.7)
for y, v in enumerate(l4.units):
    ax.text(v, y, f" {v:,}", va="center", fontsize=8, color=INK2)
ax.grid(axis="y", visible=False)
ax.set_xlabel("units, 26 Aug – 22 Sep 2020")
ax.tick_params(axis="y", labelsize=8)
ax.set_title("Top-20 styles by units, last 4 weeks")
fig.savefig(FIG / "04_top20_last4w.png")
plt.close(fig)

# %% Fig 5 — share of weekly units from styles < 12 weeks old (after a 26-week burn-in)
sw["age_w"] = (sw.week_start - sw.product_code.map(life["first"])).dt.days // 7
new_share = (sw.assign(new=sw.age_w < 12, nu=lambda d: d.units * d.new)
             .groupby("week_start")[["nu", "units"]].sum())
new_share["share"] = new_share.nu / new_share.units
ns = new_share[new_share.index >= pd.Timestamp(config.FIRST_WEEK + timedelta(weeks=26))]
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(ns.index, ns.share, color=SERIES[0], lw=2)
ax.axhline(ns.share.mean(), color=INK2, lw=1, ls="--")
ax.text(ns.index[0], ns.share.mean() + 0.015, f"mean {ns.share.mean():.0%}", color=INK2, fontsize=9)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_ylim(0, 1)
ax.set_title("Share of weekly units from styles first sold < 12 weeks earlier")
fig.savefig(FIG / "05_new_style_share.png")
plt.close(fig)

# %% Insights (computed, not hard-coded)
last52 = sw[sw.week_start >= pd.Timestamp(config.FINAL_CUTOFF - timedelta(weeks=52))]
prev52 = sw[(sw.week_start < pd.Timestamp(config.FINAL_CUTOFF - timedelta(weeks=52)))
            & (sw.week_start >= pd.Timestamp(config.FIRST_WEEK))]
by_style = last52.groupby("product_code").units.sum().sort_values(ascending=False)
cum = by_style.cumsum() / by_style.sum()
top1 = by_style.head(max(1, len(by_style) // 100)).sum() / by_style.sum()
n50 = int((cum < 0.5).sum()) + 1
tr = weeks.loc[weeks.units.idxmin()]
ly_same = weeks.set_index("week_start").units
l8 = ly_same.iloc[-8:].sum()
l8_ly = ly_same.iloc[-60:-52].sum()
online = sw.assign(on=sw.units * sw.online_share, yr=sw.week_start.dt.year).groupby("yr")[["on", "units"]].sum()
online = online.on / online.units
mix = ig.div(ig.sum(axis=1), axis=0)
swing = (mix.loc[[s for s in full_seasons if s.startswith("summer")]].mean()
         - mix.loc[[s for s in full_seasons if s.startswith("winter")]].mean()).sort_values()

insights = [
    f"**Strong seasonality.** Weekly units range from {tr.units / 1e3:,.0f}k (week of {tr.week_start:%d %b %Y}) "
    f"to {pk.units / 1e3:,.0f}k (week of {pk.week_start:%d %b %Y}), a {pk.units / tr.units:.1f}× swing; "
    f"the last 8 weeks are {l8 / l8_ly - 1:+.0%} vs the same 8 weeks a year earlier.",
    f"**Long tail.** In the last 52 weeks {len(by_style):,} styles sold at least once; the top 1% "
    f"({len(by_style) // 100:,} styles) took {top1:.0%} of units and just {n50:,} styles "
    f"({n50 / len(by_style):.1%}) made half of all units.",
    f"**Short lifecycles.** For styles launched inside the window, the median selling span is {med:.0f} weeks "
    f"(median {lf.active.median():.0f} weeks with any sale); {(lf.span <= 12).mean():.0%} are done within 12 weeks.",
    f"**Newness drives volume.** On average {ns.share.mean():.0%} of weekly units come from styles first sold "
    f"< 12 weeks earlier (range {ns.share.min():.0%}–{ns.share.max():.0%}) — history-only models will miss "
    f"much of next period's demand.",
    f"**Mix shifts with season.** Summer vs winter, {swing.index[-1]} gains {swing.iloc[-1]:+.1%} pts of unit "
    f"share and {swing.index[0]} loses {swing.iloc[0]:+.1%} pts; online share of units moved from "
    f"{online.get(2019, np.nan):.0%} (2019) to {online.get(2020, np.nan):.0%} (2020, COVID).",
]
print("\nEDA insights:")
for s in insights:
    print(" -", s.replace("**", ""))
(FIG / "eda_insights.md").write_text("# EDA insights\n\n" + "\n".join(f"- {s}" for s in insights) + "\n",
                                     encoding="utf-8")
print(f"\nSaved 5 figures + eda_insights.md to {FIG}")
