"""Take app screenshots into docs/screenshots/ with headless Chromium (dev only).

  pip install playwright && python -m playwright install chromium   # plus: sudo python -m playwright install-deps chromium
  python scripts/screenshots.py          # needs the API (port 8000) and Streamlit (port 8501) running
"""
from playwright.sync_api import sync_playwright
BASE, OUT = "http://localhost:8501", "docs/screenshots"
shots = [("top_styles", "/?page=top", "Top predicted styles", 1950),
         ("style_detail", "/?page=detail&style_id=0751471", "Why the model picked it", 2350),
         ("style_detail_richie", "/?page=detail&style_id=0685814", "Next-season concept", 2350),
         ("seasonal_view", "/?page=seasonal", "Category mix shift", 1600),
         ("model_performance", "/?page=performance", "Regressor vs baselines", 1550),
         ("concepts", "/?page=concepts", "Next-season concepts", 1500)]
with sync_playwright() as p:
    b = p.chromium.launch()
    for name, path, marker, h in shots:
        pg = b.new_page(viewport={"width": 1440, "height": h})
        pg.goto(BASE + path)
        pg.get_by_text(marker).first.wait_for(timeout=60000)
        pg.wait_for_timeout(4000)
        body = pg.inner_text("body")
        assert not any(w in body for w in ("Traceback", "StreamlitAPIException", "not reachable")), name
        pg.screenshot(path=f"{OUT}/{name}.png")
        print(name, "saved")
        pg.close()
    b.close()
