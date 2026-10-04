"""Merchmix style intelligence: Streamlit frontend. Talks only to the backend API.

  API_URL=http://localhost:8000 streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501

Pages: Top styles · Style detail · Model performance · Seasonal view · Concepts. A detail view can be linked
directly with ?style_id=0751471 (and &season=SS2020).
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

PAGES = ["Top styles", "Style detail", "Model performance", "Seasonal view", "Concepts"]
PAGE_KEYS = {"top": "Top styles", "detail": "Style detail", "performance": "Model performance",
             "seasonal": "Seasonal view", "concepts": "Concepts"}
PAGE_IDS = {v: k for k, v in PAGE_KEYS.items()}
TTL = 60


# --- API access (cached; exceptions are not cached, so a restarted API is picked up) -------------------------------
@st.cache_data(ttl=TTL, show_spinner=False)
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


def show_image(url: str | None, caption: str | None = None) -> None:
    st.image(image_bytes(url) or placeholder(), caption=caption, width="stretch")


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


def badge(text: str, ok: bool) -> None:
    color = "green" if ok else "orange"
    st.markdown(f":{color}-badge[{'✓' if ok else '!'} {text}]")


# --- navigation ---------------------------------------------------------------------------------------------------
def open_style(style_id: str, season: str | None = None) -> None:
    """Jump to the detail page. Widget state can't change after the widgets exist, so the jump is stored and
    applied at the top of the next run (before the sidebar widgets are created)."""
    st.session_state["_goto"] = {"nav": "Style detail", "style_id": style_id, "season": season}
    st.rerun()


def sidebar(seasons: list[dict]) -> tuple[str, str]:
    qp = st.query_params
    ids = [s["id"] for s in seasons]
    default = next((s["id"] for s in seasons if s["default"]), ids[0])
    if "season" not in st.session_state:
        st.session_state["season"] = qp.get("season", default).upper() if qp.get("season", "").upper() in ids else default
    if "nav" not in st.session_state:
        start = PAGE_KEYS.get(qp.get("page", ""), "Style detail" if qp.get("style_id") else "Top styles")
        st.session_state["nav"] = start
    goto = st.session_state.pop("_goto", None)
    if goto:
        st.session_state["nav"] = goto["nav"]
        if goto.get("season") in ids:
            st.session_state["season"] = goto["season"]
        st.session_state[f"pick_{st.session_state['season']}"] = goto["style_id"]
        qp["style_id"] = goto["style_id"]
    with st.sidebar:
        st.title("Merchmix")
        st.caption("Next-season style intelligence · H&M data")
        page = st.radio("Page", PAGES, key="nav")
        labels = {s["id"]: s["label"] for s in seasons}
        season = st.selectbox("Season", ids, key="season", format_func=lambda i: labels[i])
        s = next(x for x in seasons if x["id"] == season)
        st.caption(f"Cutoff {s['cutoff']} · forecast window {s['forecast_window']['start']} → "
                   f"{s['forecast_window']['end']}" + (" · actuals available" if s["observed"] else ""))
    qp["page"] = PAGE_IDS[page]
    qp["season"] = season
    if page != "Style detail" and "style_id" in qp:
        del qp["style_id"]
    return page, season


# --- pages --------------------------------------------------------------------------------------------------------
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


def page_top(season: str) -> None:
    head = api("/styles/top", limit=200, offset=0, season=season)
    st.header("Top predicted styles")
    st.caption(f"{head['season_label']} · forecast window {head['forecast_window']['start']} → "
               f"{head['forecast_window']['end']}. Ranked by forecast units; the top 3 are the highest forecast per "
               "garment group among styles still selling.")
    top3 = [s for s in head["styles"] if s["rank"]]
    cols = st.columns(len(top3) or 1)
    for col, s in zip(cols, top3):
        with col, st.container(border=True):
            st.subheader(f"#{s['rank']} {s['name']}")
            show_image(s["image_url"])
            a, b = st.columns(2)
            a.metric("Prediction score", score(s["prediction_score"]))
            b.metric("Forecast units", num(s["forecast_units"]))
            if head["observed"]:
                st.caption(f"Actual: {units(s['actual_units'])} (rank {s['actual_rank']:,})")
            st.caption(f"{s['category']['product_type']} · {s['category']['garment_group']}")
            concept = api(f"/styles/{s['style_id']}", season=season).get("concept")
            if concept:
                ok = concept["critic"]["decision"] == "approve"
                badge(concept["critic"]["status"] or ("Critic: approved" if ok else "Critic: not approved"), ok)
            if st.button("Open details", key=f"open_{s['style_id']}", width="stretch"):
                open_style(s["style_id"])

    st.subheader("All styles")
    st.caption(skipped_caption(head["styles"]))
    f1, f2 = st.columns([3, 1])
    groups = sorted({s["category"]["garment_group"] for s in head["styles"] if s["category"]["garment_group"]})
    chosen = f1.multiselect("Garment group", groups, placeholder="All garment groups")
    limit = f2.slider("Number of styles", 10, 200, 50, step=10)
    by_rank = sorted(head["styles"], key=lambda s: s["forecast_rank"])
    rows = [s for s in by_rank[:limit] if not chosen or s["category"]["garment_group"] in chosen]
    if not rows:
        st.info("No styles match the filter.")
        return
    df = pd.DataFrame([{
        "Rank": s["forecast_rank"], "Top-3 pick": "✓" if s["rank"] else "", "Photo": thumbnail_uri(s["image_url"]),
        "Style": s["style_id"], "Name": s["name"], "Prediction score": s["prediction_score"],
        "Confidence (top 1%)": s["confidence_top1pct"], "Forecast units": round(s["forecast_units"]),
        **({"Actual units": s["actual_units"]} if head["observed"] else {}),
        "Product type": s["category"]["product_type"], "Garment group": s["category"]["garment_group"]}
        for s in rows])
    event = st.dataframe(
        df, hide_index=True, width="stretch", height=min(57 * len(df) + 40, 900), row_height=56,
        on_select="rerun", selection_mode="single-row", key=f"table_{season}",
        column_config={
            "Rank": st.column_config.NumberColumn(format="%d", width=55, help="rank by forecast units"),
            "Top-3 pick": st.column_config.TextColumn(width=75, help="one of the 3 selected styles"),
            "Photo": st.column_config.ImageColumn(width=60),
            "Style": st.column_config.TextColumn(width=70),
            "Name": st.column_config.TextColumn(width=190),
            "Prediction score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f",
                                                                help="calibrated P(top 0.1% seller)"),
            "Confidence (top 1%)": st.column_config.NumberColumn("Conf. (top 1%)", format="%.2f", width=95,
                                                                 help="calibrated P(top 1% seller)"),
            "Forecast units": st.column_config.NumberColumn(format="localized", width=95),
            "Actual units": st.column_config.NumberColumn(format="localized"),
        })
    st.caption("Select a row to open the style.")
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
    names = ["Weekly units (history)", "Forecast (weekly average)", "Actual weekly units"][:len(layers)]
    top = float(data["units"].max() or 0) * 1.08 or 1
    colors = alt.Scale(domain=names, range=["#2a78d6", "#eb6834", "#1baf7a"][:len(layers)])
    base = alt.Chart(data).encode(
        x=alt.X("week_start:T", title="Week starting"),
        y=alt.Y("units:Q", title="Units per week", axis=alt.Axis(format=",d"),
                scale=alt.Scale(domain=[0, top], nice=False, clamp=False)),
        color=alt.Color("series:N", scale=colors, legend=alt.Legend(orient="bottom", title=None, labelLimit=260)),
        tooltip=[alt.Tooltip("week_start:T", title="Week"), alt.Tooltip("units:Q", format=",.0f"), "series:N"])
    line = base.mark_line(point=True).encode(strokeDash=alt.condition(
        alt.datum.series == "Forecast (weekly average)", alt.value([6, 4]), alt.value([1, 0])))
    rule = alt.Chart(pd.DataFrame({"cutoff": [cutoff]})).mark_rule(color="#52514e", strokeDash=[2, 2]).encode(x="cutoff:T")
    label = alt.Chart(pd.DataFrame({"cutoff": [cutoff], "t": ["cutoff"]})).mark_text(
        align="left", dx=4, dy=-120, color="#52514e").encode(x="cutoff:T", text="t:N")
    return (line + rule + label).properties(height=320)


def page_detail(season: str) -> None:
    head = api("/styles/top", limit=200, offset=0, season=season)
    options = {s["style_id"]: f"{'#' + str(s['rank']) + ' · ' if s['rank'] else ''}{s['name']} ({s['style_id']})"
               for s in head["styles"]}
    current = st.query_params.get("style_id") or head["styles"][0]["style_id"]
    pick = st.selectbox("Style", list(options) if current in options else [current, *options], key=f"pick_{season}",
                        index=(list(options).index(current) if current in options else 0),
                        format_func=lambda i: options.get(i, i))
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
    rank = f"#{d['rank']} (selected)" if d["rank"] else f"forecast rank #{d['forecast_rank']}"
    st.caption(f"{d['category']['product_type']} · {d['category']['garment_group']} · {rank} · "
               f"window {d['forecast_window']['start']} → {d['forecast_window']['end']}")
    st.space("small")  # keeps the image's hover toolbar (expand icon) clear of the subtitle

    left, right = st.columns([1, 2])
    with left:
        show_image(d["image_url"])
    with right:
        m = st.columns(4 + bool(d.get("actual")))
        m[0].metric("Prediction score", score(d["prediction_score"]), help="calibrated P(top 0.1% seller)")
        m[1].metric("Confidence (top 1%)", score(d["confidence_top1pct"]))
        m[2].metric("Forecast units", num(d["forecast_units"]))
        m[3].metric("Rank", f"#{d['rank']}" if d["rank"] else f"#{d['forecast_rank']} by units")
        if d.get("actual"):
            m[4].metric("Actual units", num(d["actual"]["units"]), help=f"actual rank {d['actual']['rank']:,}")
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

    st.subheader("Sales: last 26 weeks and the 4-week forecast")
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
        ok = c["critic"]["decision"] == "approve"
        badge(c["critic"]["status"] or ("Critic: approved" if ok else "Critic: not approved"), ok)
        i1, i2, txt = st.columns([1, 1, 1.4])
        with i1:
            show_image(c["reference_image_url"], "Reference (best-selling colourway)")
        with i2:
            show_image(c["image_url"], "Generated concept")
        with txt:
            st.markdown("**KEEP**")
            for t in c["keep"]:
                st.markdown(f"- {t['trait']}")
            st.markdown("**CHANGE**")
            for t in c["change"]:
                st.markdown(f"- **{t['axis']}**: {t.get('from') or '–'} → {t.get('to') or '–'}")
        if c.get("what_changed"):
            st.caption(f"What changed: {c['what_changed']}")
        if c["critic"].get("note"):
            st.caption(f"Critic note: {c['critic']['note']}")
        if c["critic"]["changes_not_visible"]:
            st.caption("Not visible in the image: " + "; ".join(c["critic"]["changes_not_visible"]))


def page_performance() -> None:
    m = api("/model/summary")
    st.header("Model performance")
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
                               for r in reg["backtest"]]), hide_index=True, width="stretch")
    st.caption(" · ".join(f"Wins on NDCG@50 vs {w['baseline']}: {w['wins']}/{w['of']}" for w in reg["ndcg50_wins"]))
    st.caption("Precision@12 = share of the predicted top 12 that are in the actual top 12; k = 12 matches the H&M "
               "Kaggle competition's MAP@12. The regressor's precision at 20 and 50 against the winner labels is in "
               "the classifier tables below (row 'LightGBM regressor (units)').")

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
                         column_config={k: st.column_config.NumberColumn(format="%.3f") for k in
                                        ("PR-AUC", "P@20", "P@50", "PR-AUC (valid.)")})
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


def page_seasonal(season: str) -> None:
    st.header("Seasonal view: SS2020 vs AW2020")
    seasons = {s["id"]: s for s in api("/seasons")["seasons"]}
    st.caption("The same pipeline at two cutoffs. SS2020 (cutoff 27 May 2020) is in the data, so predictions can be "
               "compared with what actually sold; AW2020 is the forecast. Use the sidebar to browse either season.")
    left, right = st.columns(2)
    for col, sid in ((left, "SS2020"), (right, "AW2020")):
        if sid not in seasons:
            continue
        s = seasons[sid]
        top = api("/styles/top", limit=200, season=sid)
        rows = sorted(top["styles"], key=lambda x: x["forecast_rank"])[:10]
        with col:
            st.subheader(s["label"])
            st.caption(f"Forecast window {s['forecast_window']['start']} → {s['forecast_window']['end']}")
            df = pd.DataFrame([{"#": r["forecast_rank"], "Name": r["name"], "Group": r["category"]["garment_group"],
                                "Forecast": round(r["forecast_units"]),
                                **({"Actual": r["actual_units"], "Actual rank": r["actual_rank"]} if s["observed"] else {})}
                               for r in rows])
            st.dataframe(df, hide_index=True, width="stretch",
                         column_config={"#": st.column_config.NumberColumn(width=40),
                                        "Forecast": st.column_config.NumberColumn(format="localized", width=80),
                                        "Actual": st.column_config.NumberColumn(format="localized", width=80),
                                        "Actual rank": st.column_config.NumberColumn(format="%d", width=90)})
            if s["observed"]:
                hits = sum(r["actual_rank"] <= 10 for r in rows)
                st.markdown(f"**{hits}/10** of the predicted top 10 were in the actual top 10.")
                full = pd.DataFrame([{"Forecast": r["forecast_units"], "Actual": r["actual_units"], "Name": r["name"]}
                                     for r in top["styles"] if r["actual_units"] is not None])
                mx = float(max(full["Forecast"].max(), full["Actual"].max()))
                diag = alt.Chart(pd.DataFrame({"x": [0, mx], "y": [0, mx]})).mark_line(
                    color="#52514e", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
                pts = alt.Chart(full).mark_circle(size=40, color="#2a78d6", opacity=0.7).encode(
                    x=alt.X("Forecast:Q", axis=alt.Axis(format=",d"), title="Forecast units"),
                    y=alt.Y("Actual:Q", axis=alt.Axis(format=",d"), title="Actual units"),
                    tooltip=["Name", alt.Tooltip("Forecast:Q", format=",.0f"), alt.Tooltip("Actual:Q", format=",.0f")])
                st.altair_chart((diag + pts).properties(height=300, title="Top 200: forecast vs actual"),
                                width="stretch")
            else:
                st.info("The AW2020 window (23 Sep – 20 Oct 2020) is after the end of the data, so there are no "
                        "actuals to compare.")

    mix = api("/model/summary")["seasonal"]
    st.subheader("Category mix shift (share of predicted units in each season's top 100)")
    df = pd.DataFrame(mix["category_mix"])
    long = df.melt(id_vars=["product_group", "change_pts"], value_vars=["SS2020", "AW2020"], var_name="Season",
                   value_name="Share")
    chart = alt.Chart(long).mark_bar().encode(
        y=alt.Y("product_group:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)), x=alt.X("Share:Q", axis=alt.Axis(format="%"), title="Share"),
        color=alt.Color("Season:N", scale=alt.Scale(domain=["SS2020", "AW2020"], range=["#eda100", "#2a78d6"])),
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
    top = api("/styles/top", limit=3, season="AW2020")
    cols = st.columns(3)
    for col, s in zip(cols, [x for x in top["styles"] if x["rank"]]):
        c = api(f"/styles/{s['style_id']}", season="AW2020").get("concept") or {}
        with col, st.container(border=True):
            st.markdown(f"**#{s['rank']} {s['name']}** · {s['category']['product_type']}")
            st.caption(f"{units(s['forecast_units'])} forecast · score {score(s['prediction_score'])}")
            if c:
                ok = c["critic"]["decision"] == "approve"
                badge(c["critic"]["status"] or ("Critic: approved" if ok else "Critic: not approved"), ok)
                st.markdown(c.get("what_changed") or "")
            if st.button("Open style", key=f"concept_{s['style_id']}", width="stretch"):
                open_style(s["style_id"], season="AW2020")


def main() -> None:
    st.set_page_config(page_title="Merchmix style intelligence", page_icon="👗", layout="wide")
    try:
        seasons = api("/seasons")["seasons"]
        page, season = sidebar(seasons)
        if page == "Top styles":
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
