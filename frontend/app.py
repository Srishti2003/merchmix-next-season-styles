"""Merchmix style intelligence: Streamlit frontend. Talks only to the backend API.

  API_URL=http://localhost:8000 streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501

Pages: Overview · Top styles · Style detail · Model performance · Seasonal view · Concepts. A detail view can be
linked directly with ?style_id=0751471 (and &season=SS2020).
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontend import api_client  # noqa: E402
from frontend.api_client import ApiError, ApiUnavailable  # noqa: E402

PAGES = ["Overview", "Top styles", "Style detail", "Model performance", "Seasonal view", "Concepts"]
PAGE_KEYS = {"overview": "Overview", "top": "Top styles", "detail": "Style detail", "performance": "Model performance",
             "seasonal": "Seasonal view", "concepts": "Concepts"}
PAGE_IDS = {v: k for k, v in PAGE_KEYS.items()}
TTL = 60
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"

HELP = {
    "score": "Prediction score: the calibrated chance that the style is a top-0.1% seller (about the best 20 styles) "
             "over the 4-week forecast window.",
    "top1": "Calibrated chance that the style is a top-1% seller (about the best 200 styles) over the 4-week window.",
    "units": "Expected units sold over the 4-week forecast window (the regressor's forecast).",
    "rank": "Position among all scored styles when sorted by forecast units.",
    "pick": "One of the 3 selected styles: the highest forecast in its garment group among styles still selling.",
    "actual": "Units actually sold in the forecast window (only for seasons whose window is inside the data).",
    "scored": "Styles with sales in the 12 weeks before the cutoff, all scored by the model.",
    "window": "The 4 weeks the forecast covers, starting at the cutoff.",
    "top3": "Sum of the forecast units of the 3 selected styles.",
    "vs_baseline": "Backtest weeks (out of 10) where the model ranked the actual best sellers better than repeating "
                   "last week's sales × 4 (NDCG@50).",
}


# --- API access (cached; exceptions are not cached, so a restarted API is picked up) -------------------------------
@st.cache_data(ttl=TTL, show_spinner="Loading from the API…")
def api(path: str, **params) -> dict:
    return api_client.get_json(path, {k: v for k, v in params.items() if v is not None} or None)


@st.cache_data(ttl=TTL, show_spinner=False)
def image_bytes(url: str | None) -> bytes | None:
    return api_client.get_bytes(url)


@st.cache_data(show_spinner=False)
def placeholder(w: int = 300, h: int = 400, text: str = "No photo") -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (w, h), (236, 235, 231))
    d = ImageDraw.Draw(img)
    tw = d.textlength(text)
    d.text(((w - tw) / 2, h / 2 - 6), text, fill=(120, 119, 115))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data(ttl=TTL, show_spinner=False)
def thumbnail_uri(url: str | None, height: int = 96) -> str:
    """Small JPEG data URI for table thumbnails (never a raw API URL: the browser can't reach the API)."""
    from PIL import Image

    raw = image_bytes(url) or placeholder(72, 96)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((height, height))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def show_image(url: str | None, caption: str | None = None, width: int | str = "stretch") -> None:
    st.image(image_bytes(url) or placeholder(), caption=caption, width=width)


def api_down(e: Exception) -> None:
    st.error(f"**The API is not reachable.** {e}\n\nStart it in a terminal from the repository root:")
    st.code(api_client.RUN_API, language="bash")
    st.caption(f"The frontend uses API_URL = {api_client.API_URL} (set the API_URL environment variable to change it).")
    st.stop()


# --- formatting ---------------------------------------------------------------------------------------------------
def units(x) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:,.0f} units"


def num(x) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:,.0f}"


def score(x) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:.2f}"


def pick_label(s: dict) -> str:
    """'Pick #3 · rank 4 by units' for a selected style, 'Rank 5 by units' otherwise."""
    by_units = f"rank {s['forecast_rank']} by units"
    return f"Pick #{s['rank']} · {by_units}" if s["rank"] else by_units[:1].upper() + by_units[1:]


def badge(text: str, ok: bool) -> None:
    color = "green" if ok else "orange"
    st.markdown(f":{color}-badge[{'✓' if ok else '!'} {text}]")


def critic_badge(concept: dict) -> None:
    ok = concept["critic"]["decision"] == "approve"
    badge(concept["critic"]["status"] or ("Critic: approved" if ok else "Critic: not approved"), ok)


def window_text(w: dict) -> str:
    start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
    return f"{start:%-d %b} → {end:%-d %b %Y}"


# --- navigation ---------------------------------------------------------------------------------------------------
def go(page: str, style_id: str | None = None, season: str | None = None) -> None:
    """Jump to a page (and style). Widget state can't change after the widgets exist, so the jump is stored and
    applied at the top of the next run (before the sidebar widgets are created)."""
    st.session_state["_goto"] = {"nav": page, "style_id": style_id, "season": season}
    st.rerun()


def open_style(style_id: str, season: str | None = None) -> None:
    go("Style detail", style_id, season)


def sidebar(seasons: list[dict]) -> tuple[str, str]:
    qp = st.query_params
    ids = [s["id"] for s in seasons]
    default = next((s["id"] for s in seasons if s["default"]), ids[0])
    if "season" not in st.session_state:
        st.session_state["season"] = qp.get("season", default).upper() if qp.get("season", "").upper() in ids else default
    if "nav" not in st.session_state:
        start = PAGE_KEYS.get(qp.get("page", ""), "Style detail" if qp.get("style_id") else "Overview")
        st.session_state["nav"] = start
    goto = st.session_state.pop("_goto", None)
    if goto:
        st.session_state["nav"] = goto["nav"]
        if goto.get("season") in ids:
            st.session_state["season"] = goto["season"]
        if goto.get("style_id"):
            st.session_state[f"pick_{st.session_state['season']}"] = goto["style_id"]
            qp["style_id"] = goto["style_id"]
    with st.sidebar:
        st.title("Merchmix")
        st.caption("Next-season style intelligence · H&M data")
        page = st.radio("Page", PAGES, key="nav", label_visibility="collapsed")
        labels = {s["id"]: f"{s['id']} · {'backtest' if s['observed'] else 'forecast'}" for s in seasons}
        season = st.selectbox("Season", ids, key="season", format_func=lambda i: labels[i])
        s = next(x for x in seasons if x["id"] == season)
        st.caption(f"Cutoff {s['cutoff']} · window {window_text(s['forecast_window'])}"
                   + (" · actuals available" if s["observed"] else ""))
    qp["page"] = PAGE_IDS[page]
    qp["season"] = season
    if page != "Style detail" and "style_id" in qp:
        del qp["style_id"]
    return page, season


# --- pages --------------------------------------------------------------------------------------------------------
def page_overview(season: str) -> None:
    head = api("/styles/top", limit=200, offset=0, season=season)
    summary = api("/model/summary")
    reg = summary["regressor"]
    top3 = [s for s in head["styles"] if s["rank"]]
    base = next((w for w in reg["ndcg50_wins"] if w["baseline"] == "Last week × 4"), None)
    model_bt = next((r for r in reg["backtest"] if r["method"] == "LightGBM regressor"), None)
    base_bt = next((r for r in reg["backtest"] if r["method"] == "Last week × 4"), None)

    st.header("Overview")
    st.caption(f"{head['season_label']}: which styles will sell best in the next 4 weeks, and why.")
    w = head["forecast_window"]
    k = st.columns([1, 1.6, 1, 1])
    k[0].metric("Styles scored", num(head["n_styles_scored"]), help=HELP["scored"], border=True)
    k[1].metric(f"Forecast window ({pd.Timestamp(w['end']):%Y})",
                f"{pd.Timestamp(w['start']):%-d %b}–{pd.Timestamp(w['end']):%-d %b}", help=HELP["window"], border=True)
    k[2].metric("Top-3 forecast units", num(sum(s["forecast_units"] for s in top3)), help=HELP["top3"], border=True)
    if base:
        k[3].metric("Beats last week × 4", f"{base['wins']}/{base['of']}", help=HELP["vs_baseline"],
                    border=True)
    if base and model_bt and base_bt:
        st.markdown(f"**The model ranks the coming best sellers better than repeating last week's sales in "
                    f"{base['wins']} of {base['of']} backtest weeks** (NDCG@50 {model_bt['ndcg@50']['mean']:.3f} vs "
                    f"{base_bt['ndcg@50']['mean']:.3f}).")

    st.subheader("The 3 picks")
    for col, s in zip(st.columns(3), top3):
        with col, st.container(border=True):
            img, txt = st.columns([1, 2])
            with img:
                show_image(s["image_url"])
            with txt:
                st.caption(pick_label(s))
                st.markdown(f"**{s['name']}**")
                st.caption(f"{s['category']['product_type']} · {s['category']['garment_group']}")
                st.markdown(f"{num(s['forecast_units'])} units · score {score(s['prediction_score'])}")
            if st.button("Open style →", key=f"ov_{s['style_id']}", width="stretch"):
                open_style(s["style_id"])

    with st.container(border=True):
        st.markdown("**How to read this**")
        st.markdown(
            "- **Prediction score**: the chance (0–1) that a style is a top-0.1% seller, about the best 20 styles, "
            "over the next 4 weeks.\n"
            "- **Forecast units**: expected units sold over those 4 weeks.\n"
            "- **Chance of top 1%**: the chance of being in the best 1% (about 200 styles); most top-50 styles are "
            "close to 1.\n"
            "- **Pick #1–3**: the highest forecast in each garment group among styles still selling, so the picks "
            "cover different kinds of garment.")

    st.subheader("Explore")
    links = (("Top styles", "Search, filter and download every scored style in the top 200."),
             ("Model performance", "How the model compares with simple rules of thumb."),
             ("Seasonal view", "SS2020 forecast vs what actually sold, and the category mix shift."),
             ("Concepts", "The next-season design concept for each pick."))
    for col, (page, text) in zip(st.columns(4), links):
        with col:
            if st.button(page, key=f"link_{PAGE_IDS[page]}", width="stretch"):
                go(page)
            st.caption(text)


def skipped_caption(styles: list[dict]) -> str:
    """Explain the per-garment-group rule with the highest-ranked style it skipped."""
    rule = "Top 3 = highest forecast per garment group"
    picks = {s["category"]["garment_group"]: s for s in styles if s["rank"]}
    last = max((s["forecast_rank"] for s in picks.values()), default=0)
    skipped = next((s for s in sorted(styles, key=lambda s: s["forecast_rank"])
                    if not s["rank"] and s["forecast_rank"] < last and s["category"]["garment_group"] in picks), None)
    if skipped is None:
        return rule + "."
    group = skipped["category"]["garment_group"]
    return (f"{rule}, so {skipped['name']} (#{skipped['forecast_rank']} by units) is skipped because {group} is "
            f"already covered by {picks[group]['name']}.")


def filter_styles(styles: list[dict], query: str = "", index_groups=(), product_types=(), garment_groups=(),
                  top3_only: bool = False, sort: str = "Forecast units") -> list[dict]:
    """Search (name or style id), filter and sort the style list."""
    q = query.strip().lower()

    def matches(s: dict) -> bool:
        names = f"{s['name'] or ''} {s.get('raw_name') or ''}".lower()
        return not q or q in names or (q.isdigit() and q.lstrip("0") in s["style_id"].lstrip("0"))

    out = [s for s in styles if matches(s)
           and (not index_groups or s["category"].get("index_group") in index_groups)
           and (not product_types or s["category"]["product_type"] in product_types)
           and (not garment_groups or s["category"]["garment_group"] in garment_groups)
           and (not top3_only or s["rank"])]
    if sort == "Prediction score":
        return sorted(out, key=lambda s: (-s["prediction_score"], s["forecast_rank"]))
    return sorted(out, key=lambda s: s["forecast_rank"])


def page_top(season: str) -> None:
    head = api("/styles/top", limit=200, offset=0, season=season)
    st.header("Top predicted styles")
    st.caption(f"{head['season_label']} · forecast window {window_text(head['forecast_window'])}. Ranked by forecast "
               "units; the top 3 are the highest forecast per garment group among styles still selling.")
    top3 = [s for s in head["styles"] if s["rank"]]
    for col, s in zip(st.columns(len(top3) or 1), top3):
        with col, st.container(border=True):
            st.caption(pick_label(s), help=HELP["rank"])
            st.subheader(s["name"])
            show_image(s["image_url"])
            a, b = st.columns(2)
            a.metric("Prediction score", score(s["prediction_score"]), help=HELP["score"])
            b.metric("Forecast units", num(s["forecast_units"]), help=HELP["units"])
            if head["observed"]:
                st.caption(f"Actual: {units(s['actual_units'])} (rank {s['actual_rank']:,})")
            st.caption(f"{s['category']['product_type']} · {s['category']['garment_group']}")
            concept = api(f"/styles/{s['style_id']}", season=season).get("concept")
            if concept:
                critic_badge(concept)
            if st.button("Open details", key=f"open_{s['style_id']}", width="stretch"):
                open_style(s["style_id"])

    st.subheader("All styles")
    st.caption(skipped_caption(head["styles"]))
    styles = head["styles"]

    def options(key: str) -> list[str]:
        return sorted({s["category"].get(key) for s in styles if s["category"].get(key)})

    r1 = st.columns([3, 1.3, 1])
    query = r1[0].text_input("Search", placeholder="Name or style id, e.g. Pluto or 0751471", key=f"q_{season}")
    sort = r1[1].selectbox("Sort by", ["Forecast units", "Prediction score"], key=f"sort_{season}")
    with r1[2]:
        st.space("small")
        top3_only = st.toggle("Top-3 picks only", key=f"top3_{season}", help=HELP["pick"])
    r2 = st.columns(3)
    index_groups = r2[0].multiselect("Index group", options("index_group"), placeholder="All index groups",
                                     key=f"fi_{season}")
    product_types = r2[1].multiselect("Product type", options("product_type"), placeholder="All product types",
                                      key=f"fp_{season}")
    garment_groups = r2[2].multiselect("Garment group", options("garment_group"), placeholder="All garment groups",
                                       key=f"fg_{season}")
    rows = filter_styles(styles, query, index_groups, product_types, garment_groups, top3_only, sort)
    if not rows:
        st.info("No styles match the search and filters. Clear the search box or remove a filter to see more styles.")
        return
    shown = rows[:st.session_state.get(f"rows_{season}", 50)]
    df = pd.DataFrame([{
        "Rank": s["forecast_rank"], "Top-3 pick": "✓" if s["rank"] else "", "Photo": thumbnail_uri(s["image_url"]),
        "Style": s["style_id"], "Name": s["name"], "Prediction score": s["prediction_score"],
        "Chance of top 1%": s["confidence_top1pct"], "Forecast units": round(s["forecast_units"]),
        **({"Actual units": s["actual_units"]} if head["observed"] else {}),
        "Index group": s["category"].get("index_group"), "Product type": s["category"]["product_type"],
        "Garment group": s["category"]["garment_group"]}
        for s in shown])
    event = st.dataframe(
        df, hide_index=True, width="stretch", height=min(57 * len(df) + 40, 900), row_height=56,
        on_select="rerun", selection_mode="single-row", key=f"table_{season}",
        column_config={
            "Rank": st.column_config.NumberColumn(format="%d", width=55, help=HELP["rank"]),
            "Top-3 pick": st.column_config.TextColumn(width=85, help=HELP["pick"]),
            "Photo": st.column_config.ImageColumn(width=60),
            "Style": st.column_config.TextColumn(width=70, help="product_code: all colourways of one design"),
            "Name": st.column_config.TextColumn(width=200),
            "Prediction score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f",
                                                                color=BLUE, help=HELP["score"]),
            "Chance of top 1%": st.column_config.NumberColumn(format="%.2f", width=125, help=HELP["top1"]),
            "Forecast units": st.column_config.NumberColumn(format="localized", width=110, help=HELP["units"]),
            "Actual units": st.column_config.NumberColumn(format="localized", help=HELP["actual"]),
        })
    f1, f2, f3 = st.columns([2, 1, 1])
    f1.caption(f"Showing {len(df)} of {len(rows)} matching styles. Select a row to open the style.")
    f2.selectbox("Rows to show", [25, 50, 100, 200], index=1, key=f"rows_{season}", label_visibility="collapsed",
                 format_func=lambda n: f"Show {n} rows")
    f3.download_button("Download CSV", df.drop(columns="Photo").to_csv(index=False).encode("utf-8"),
                       file_name=f"merchmix_styles_{season}.csv", mime="text/csv", width="stretch",
                       help="The filtered table as shown, without photos.")
    if event and event.selection.rows:
        open_style(df.iloc[event.selection.rows[0]]["Style"])


def sales_chart(d: dict) -> alt.Chart:
    hist = pd.DataFrame(d["sales_history_26w"])
    hist["week_start"] = pd.to_datetime(hist["week_start"])
    hist["series"] = "Weekly units (history)"
    cutoff = pd.Timestamp(d["cutoff"])
    fc = pd.DataFrame({"week_start": pd.date_range(cutoff, periods=4, freq="7D"),
                       "units": d["forecast_units"] / 4, "series": "Forecast (weekly average)"})
    layers = [hist, fc]
    if d.get("actual"):
        act = pd.DataFrame(d["actual"]["weekly"])
        act["week_start"] = pd.to_datetime(act["week_start"])
        act["series"] = "Actual weekly units"
        layers.append(act)
    data = pd.concat(layers, ignore_index=True)
    top = float(data["units"].max() or 0) * 1.08 or 1
    names = ["Weekly units (history)", "Forecast (weekly average)", "Actual weekly units"][:len(layers)]
    colors = alt.Scale(domain=names, range=[BLUE, ORANGE, GREEN][:len(layers)])
    base = alt.Chart(data).encode(
        x=alt.X("week_start:T", title="Week starting"),
        y=alt.Y("units:Q", title="Units per week", axis=alt.Axis(format=",d"),
                scale=alt.Scale(domain=[0, top], nice=False, clamp=False)),
        color=alt.Color("series:N", scale=colors, legend=alt.Legend(orient="bottom", title=None, labelLimit=260)),
        tooltip=[alt.Tooltip("week_start:T", title="Week"), alt.Tooltip("units:Q", format=",.0f"), "series:N"])
    line = base.mark_line(point=True).encode(strokeDash=alt.condition(
        alt.datum.series == "Forecast (weekly average)", alt.value([6, 4]), alt.value([1, 0])))
    rule = alt.Chart(pd.DataFrame({"cutoff": [cutoff]})).mark_rule(color=GREY, strokeDash=[2, 2]).encode(x="cutoff:T")
    label = alt.Chart(pd.DataFrame({"cutoff": [cutoff], "t": ["cutoff"]})).mark_text(
        align="left", dx=4, dy=-120, color=GREY).encode(x="cutoff:T", text="t:N")
    return (line + rule + label).properties(height=320)


def page_detail(season: str) -> None:
    head = api("/styles/top", limit=200, offset=0, season=season)
    ordered = sorted(head["styles"], key=lambda s: s["forecast_rank"])
    options = {s["style_id"]: f"{'Pick #' + str(s['rank']) + ' · ' if s['rank'] else ''}{s['name']} ({s['style_id']})"
               for s in ordered}
    current = st.query_params.get("style_id") or ordered[0]["style_id"]
    current = current.zfill(7) if current.isdigit() else current  # ?style_id=751471 works too
    ids = list(options)
    pos = ids.index(current) if current in options else None

    nav = st.columns([1.2, 1, 1, 4])
    if nav[0].button("← Back to list", key="back", width="stretch"):
        go("Top styles")
    if nav[1].button("‹ Previous", key="prev", width="stretch", disabled=not pos):
        open_style(ids[pos - 1])
    if nav[2].button("Next ›", key="next", width="stretch", disabled=pos is None or pos >= len(ids) - 1):
        open_style(ids[pos + 1])

    pick = st.selectbox("Style", ids if current in options else [current, *ids], key=f"pick_{season}",
                        index=pos if pos is not None else 0, format_func=lambda i: options.get(i, i),
                        help="Styles in order of forecast units")
    if pick != st.query_params.get("style_id"):
        st.query_params["style_id"] = pick
    try:
        d = api(f"/styles/{pick}", season=season)
    except ApiError as e:
        if e.status == 404:
            st.warning(f"Style not found: {e.detail}")
            return
        raise
    st.header(f"{d['name']}  ·  {d['style_id']}")
    st.caption(f"{pick_label(d)} · {d['category']['product_type']} · {d['category']['garment_group']} · "
               f"window {window_text(d['forecast_window'])}")
    st.space("small")  # keeps the image's hover toolbar (expand icon) clear of the subtitle

    left, right = st.columns([1, 2])
    with left:
        show_image(d["image_url"])
    with right:
        act = d.get("actual")
        m = st.columns(4)
        m[0].metric("Prediction score", score(d["prediction_score"]), help=HELP["score"])
        m[1].metric("Chance of top 1%", score(d["confidence_top1pct"]), help=HELP["top1"])
        m[2].metric("Forecast units", num(d["forecast_units"]), help=HELP["units"])
        m[3].metric("Rank by units", f"#{d['forecast_rank']}", help=HELP["rank"])
        if act:  # observed season: actuals next to the forecast
            n = st.columns(4)
            n[2].metric("Actual units", num(act["units"]), delta=f"{act['units'] / d['forecast_units'] - 1:+.0%} vs forecast",
                        delta_color="off", help=HELP["actual"])
            n[3].metric("Actual rank", f"#{act['rank']:,}", help="Position by units actually sold in the window.")
        a = d["attributes"]
        p = d["performance"]
        info = pd.DataFrame([
            ("Product group", a.get("product_group_name")), ("Garment group", a.get("garment_group_name")),
            ("Index group", a.get("index_group_name")), ("Section", a.get("section_name")),
            ("Main colour", a.get("colour_group_name")), ("Appearance", a.get("graphical_appearance_name")),
            ("Colourways", f"{a.get('n_colours')}"), ("Units last week", units(p.get("units_last_1w"))),
            ("Units last 4 weeks", units(p.get("units_last_4w"))),
            ("Same 4 weeks last year", units(p.get("units_same_4w_last_year"))),
            ("Units to date", units(p.get("units_to_date"))),
            ("Weeks since launch", f"{p.get('weeks_since_launch')}"),
            ("Discount vs peak price", f"{p['discount_vs_peak_price']:.0%}" if p.get("discount_vs_peak_price") is not None else "–"),
        ], columns=["Attribute", "Value"])
        st.dataframe(info, hide_index=True, width="stretch", height=35 * len(info) + 38)
        if a.get("detail_desc"):
            st.caption(a["detail_desc"])

    st.subheader("Sales: last 26 weeks, the 4-week forecast" + (" and what actually sold" if act else ""))
    st.altair_chart(sales_chart(d), width="stretch")

    st.subheader("Why the model picked it")
    st.markdown(f"**{d['explanation']['why_selected']}**")
    for r in d["explanation"]["reasons"]:
        arrow = "▲" if r["direction"] == "raises" else "▼" if r["direction"] == "lowers" else "•"
        st.markdown(f"{arrow} {r['text']}")
    st.markdown("Big recent weeks are partly discounted, since sales tend to fall back after a spike.")
    st.caption("Drivers are the regressor's top-5 SHAP contributions, each a multiplier on 'last week × 4'.")

    c = d.get("concept")
    if c:
        st.subheader("Next-season concept")
        with st.container(border=True):
            critic_badge(c)
            i1, i2, txt = st.columns([1, 1, 1.4])
            with i1:
                show_image(c["reference_image_url"], "Reference (best-selling colourway)")
            with i2:
                show_image(c["image_url"], "Generated concept")
            with txt:
                st.caption("KEEP · what made it sell")
                st.markdown("\n".join(f"- {t['trait']}" for t in c["keep"]))
                st.caption("CHANGE · what is new")
                st.markdown("\n".join(f"- **{t['axis']}**: {t.get('from') or '–'} → {t.get('to') or '–'}"
                                      for t in c["change"]))
            if c.get("what_changed"):
                st.caption(f"What changed: {c['what_changed']}")
            if c["critic"].get("note"):
                st.caption(f"Critic note: {c['critic']['note']}")
            if c["critic"]["changes_not_visible"]:
                st.caption("Not visible in the image: " + "; ".join(c["critic"]["changes_not_visible"]))


def merchandiser_summary(m: dict) -> list[str]:
    reg, clf = m["regressor"], m["classifier_top0.1pct"]
    bt = {r["method"]: r for r in reg["backtest"]}
    wins = next((w for w in reg["ndcg50_wins"] if w["baseline"] == "Last week × 4"), None)
    methods = {r["method"]: r["backtest"] for r in clf["methods"]}
    out = []
    if wins and "LightGBM regressor" in bt and "Last week × 4" in bt:
        out.append(f"**The ranking beats the simple rule, modestly.** Ordering styles by forecast units put the real "
                   f"best sellers higher than repeating last week's sales in {wins['wins']} of {wins['of']} backtest "
                   f"weeks (NDCG@50 {bt['LightGBM regressor']['ndcg@50']['mean']:.3f} vs "
                   f"{bt['Last week × 4']['ndcg@50']['mean']:.3f}). Use the list to decide where to look first.")
    if "LightGBM classifier" in methods and "Last week × 4" in methods:
        p20, b20 = methods["LightGBM classifier"]["precision@20"], methods["Last week × 4"]["precision@20"]
        out.append(f"**About {round(p20 * 20)} of the 20 highest prediction scores turned out to be top-0.1% "
                   f"sellers** (precision@20 {p20:.3f}, vs {b20:.3f} for last week × 4).")
    out.append("**The score is a calibrated chance**, so 0.25 means roughly 1 in 4 such styles become a top-0.1% "
               "seller. A big unit forecast with a low score is worth a stock and trend check before buying deep "
               "(see Seasonal view).")
    return out


def page_performance() -> None:
    m = api("/model/summary")
    st.header("Model performance")
    with st.container(border=True):
        st.markdown("**What this means for a merchandiser**")
        st.markdown("\n".join(f"- {b}" for b in merchandiser_summary(m)))
    a, b, c = st.columns(3)
    a.markdown(f"**Success definition.** {m['success_definition']}")
    b.markdown(f"**Why 4 weeks.** {m['horizon']}")
    c.markdown(f"**Stock caveat.** {m['stock_caveat']}")
    st.caption(m["ranking"])

    reg = m["regressor"]
    st.subheader(f"Regressor vs baselines ({reg['backtest_cutoffs']}-cutoff rolling backtest)")
    st.dataframe(pd.DataFrame([{"Method": r["method"],
                                "NDCG@50": f"{r['ndcg@50']['mean']:.3f} ± {r['ndcg@50']['std']:.3f}",
                                "Precision@12": f"{r['precision@12']['mean']:.3f} ± {r['precision@12']['std']:.3f}"}
                               for r in reg["backtest"]]), hide_index=True, width="stretch",
                 column_config={"NDCG@50": st.column_config.TextColumn(
                                    help="How well the top 50 of the list matches the real best sellers (1 = perfect)"),
                                "Precision@12": st.column_config.TextColumn(
                                    help="Share of the predicted top 12 that are in the actual top 12")})
    st.caption(" · ".join(f"Wins on NDCG@50 vs {w['baseline']}: {w['wins']}/{w['of']}" for w in reg["ndcg50_wins"]))
    st.caption("@12 follows the H&M Kaggle competition's MAP@12 convention; the regressor's P@20/@50 are in the "
               "classifier tables below.")

    for key, title in (("classifier_top1pct", "Winner classifier, top 1% label"),
                       ("classifier_top0.1pct", "Winner classifier, top 0.1% label (the displayed score)")):
        clf = m[key]
        st.subheader(title)
        t, p = st.columns([3, 2])
        with t:
            st.dataframe(pd.DataFrame([{"Method": r["method"], "PR-AUC": r["backtest"]["pr_auc"],
                                        "P@20": r["backtest"]["precision@20"], "P@50": r["backtest"]["precision@50"],
                                        "PR-AUC (valid.)": r["validation"]["pr_auc"]}
                                       for r in clf["methods"]]),
                         hide_index=True, width="stretch",
                         column_config={
                             "PR-AUC": st.column_config.NumberColumn(
                                 format="%.3f", help="How well the score separates winners from the rest (1 = perfect)"),
                             "P@20": st.column_config.NumberColumn(format="%.3f",
                                                                   help="Share of the top 20 that were winners"),
                             "P@50": st.column_config.NumberColumn(format="%.3f",
                                                                   help="Share of the top 50 that were winners"),
                             "PR-AUC (valid.)": st.column_config.NumberColumn(
                                 format="%.3f", help="PR-AUC on the validation week (cutoff 2020-08-26)")})
            st.caption("PR-AUC, P@20 and P@50 (precision at 20/50) are 10-cutoff backtest means; "
                       "valid. = the validation week (cutoff 2020-08-26).")
            cal = clf["calibration"]
            st.caption(f"About {clf['winners_per_cutoff']:.0f} winners per cutoff. Classifier ahead of last week × 4 "
                       f"on PR-AUC at {clf['pr_auc_wins_vs_last_week_x4']}/{clf['backtest_cutoffs']} cutoffs. "
                       f"Calibration (isotonic, earlier folds only), mean of per-cutoff values: ECE {cal['ece_raw']:.5f} → "
                       f"{cal['ece_calibrated']:.5f}, Brier {cal['brier_raw']:.5f} → {cal['brier_calibrated']:.5f}. "
                       "The plot legend shows ECE pooled over all 11 cutoffs.")
        with p:
            show_image(clf.get("reliability_plot_url"), "Reliability: raw vs calibrated")


def risk_flag(styles: list[dict], n: int = 20, threshold: float = 0.3) -> str | None:
    """Observed season: do low prediction scores among the largest forecasts mark the big overpredictions?"""
    big = sorted([s for s in styles if s["actual_units"] is not None], key=lambda s: s["forecast_rank"])[:n]
    low = [s for s in big if s["prediction_score"] < threshold]
    high = [s for s in big if s["prediction_score"] >= threshold]
    miss = [s for s in big if s["actual_units"] < 0.5 * s["forecast_units"]]
    if not miss or not high or any(s not in low for s in miss):
        return None
    worst = max(miss, key=lambda s: s["forecast_units"] - s["actual_units"])  # the largest miss in units
    floor = int(min(s["actual_units"] / s["forecast_units"] for s in high) * 100)
    return (f"**Risk flag.** Of the {n} largest forecasts, the {len(low)} with a prediction score below {threshold} "
            f"include all {len(miss)} that sold under half their forecast (e.g. {worst['name']}: "
            f"{num(worst['forecast_units'])} forecast, {num(worst['actual_units'])} sold, score "
            f"{score(worst['prediction_score'])}); the other {len(high)} sold at least {floor}% of forecast. "
            f"One season and {len(miss)} cases, so read a low score on a big forecast as a prompt to check stock and "
            "trend, not as a correction.")


def page_seasonal(season: str) -> None:
    st.header("Seasonal view: SS2020 vs AW2020")
    seasons = {s["id"]: s for s in api("/seasons")["seasons"]}
    st.caption("The same pipeline at two cutoffs. SS2020 (cutoff 27 May 2020) is in the data, so predictions can be "
               "compared with what actually sold; AW2020 is the forecast. Use the sidebar to browse either season.")
    for sid in ("SS2020", "AW2020"):
        if sid not in seasons:
            continue
        s = seasons[sid]
        top = api("/styles/top", limit=200, season=sid)
        rows = sorted(top["styles"], key=lambda x: x["forecast_rank"])[:10]
        st.subheader(s["label"])
        st.caption(f"Forecast window {window_text(s['forecast_window'])} · top 10 by forecast units")
        df = pd.DataFrame([{"#": r["forecast_rank"], "Name": r["name"], "Group": r["category"]["garment_group"],
                            "Forecast": round(r["forecast_units"]),
                            **({"Actual": r["actual_units"], "Actual rank": r["actual_rank"],
                                "In actual top 10": "✓" if r["actual_rank"] <= 10 else "✗"} if s["observed"] else {})}
                           for r in rows])
        st.dataframe(df, hide_index=True, width="stretch",
                     column_config={"#": st.column_config.NumberColumn(width=40, help=HELP["rank"]),
                                    "Forecast": st.column_config.NumberColumn(format="localized", help=HELP["units"]),
                                    "Actual": st.column_config.NumberColumn(format="localized", help=HELP["actual"]),
                                    "Actual rank": st.column_config.NumberColumn(format="%d"),
                                    "In actual top 10": st.column_config.TextColumn(
                                        help="Was the style in the 10 best sellers of the window?")})
        if not s["observed"]:
            st.info("The AW2020 window (23 Sep – 20 Oct 2020) is after the end of the data, so there are no "
                    "actuals to compare.")
            continue
        notes, chart = st.columns([2, 3])
        with notes:
            hits = sum(r["actual_rank"] <= 10 for r in rows)
            st.markdown(f"**{hits}/10** of the predicted top 10 were in the actual top 10.")
            flag = risk_flag(top["styles"])
            if flag:
                st.markdown(flag)
        with chart:
            full = pd.DataFrame([{"Forecast": r["forecast_units"], "Actual": r["actual_units"], "Name": r["name"]}
                                 for r in top["styles"] if r["actual_units"] is not None])
            mx = float(max(full["Forecast"].max(), full["Actual"].max()))
            diag = alt.Chart(pd.DataFrame({"x": [0, mx], "y": [0, mx]})).mark_line(
                color=GREY, strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
            pts = alt.Chart(full).mark_circle(size=40, color=BLUE, opacity=0.7).encode(
                x=alt.X("Forecast:Q", axis=alt.Axis(format=",d"), title="Forecast units"),
                y=alt.Y("Actual:Q", axis=alt.Axis(format=",d"), title="Actual units"),
                tooltip=["Name", alt.Tooltip("Forecast:Q", format=",.0f"), alt.Tooltip("Actual:Q", format=",.0f")])
            st.altair_chart((diag + pts).properties(height=300, title="Top 200: forecast vs actual"),
                            width="stretch")

    mix = api("/model/summary")["seasonal"]
    st.subheader("Category mix shift (share of predicted units in each season's top 100)")
    df = pd.DataFrame(mix["category_mix"])
    long = df.melt(id_vars=["product_group", "change_pts"], value_vars=["SS2020", "AW2020"], var_name="Season",
                   value_name="Share")
    chart = alt.Chart(long).mark_bar().encode(
        y=alt.Y("product_group:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)), x=alt.X("Share:Q", axis=alt.Axis(format="%"), title="Share"),
        color=alt.Color("Season:N", scale=alt.Scale(domain=["SS2020", "AW2020"], range=["#eda100", BLUE])),
        yOffset="Season:N", tooltip=["product_group", "Season", alt.Tooltip("Share:Q", format=".0%")])
    st.altair_chart(chart.properties(height=360), width="stretch")
    if mix.get("what_changed"):
        text = mix["what_changed"].replace("summer 2020", "SS2020").replace("autumn 2020", "AW2020")
        st.caption(text[:1].upper() + text[1:])


def page_concepts() -> None:
    st.header("Next-season concepts")
    st.caption("One generated concept per predicted AW2020 winner: it keeps what made the style sell and changes "
               "fabric, colour or trims. No new images were generated for this app.")
    show_image("/images/generated_concepts.png")
    top = api("/styles/top", limit=200, season="AW2020")
    cols = st.columns(3)
    for col, s in zip(cols, [x for x in top["styles"] if x["rank"]]):
        c = api(f"/styles/{s['style_id']}", season="AW2020").get("concept") or {}
        with col, st.container(border=True):
            st.caption(pick_label(s))
            st.markdown(f"**{s['name']}** · {s['category']['product_type']}")
            st.caption(f"{units(s['forecast_units'])} forecast · score {score(s['prediction_score'])}")
            if c:
                critic_badge(c)
                st.markdown(c.get("what_changed") or "")
            if st.button("Open style", key=f"concept_{s['style_id']}", width="stretch"):
                open_style(s["style_id"], season="AW2020")


def main() -> None:
    st.set_page_config(page_title="Merchmix style intelligence", page_icon="👗", layout="wide")
    try:
        seasons = api("/seasons")["seasons"]
        page, season = sidebar(seasons)
        if page == "Overview":
            page_overview(season)
        elif page == "Top styles":
            page_top(season)
        elif page == "Style detail":
            page_detail(season)
        elif page == "Model performance":
            page_performance()
        elif page == "Seasonal view":
            page_seasonal(season)
        else:
            page_concepts()
    except ApiUnavailable as e:
        api_down(e)
    except ApiError as e:
        st.warning(f"The API returned an error ({e.status}): {e.detail}")


main()
