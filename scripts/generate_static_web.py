"""
Generate static HTML for viewing heatmaps without Flask.

Run from stock_analysis/:  python scripts/generate_static_web.py
Then open heatmaps/index.html in your browser, or run: python -m http.server 8000 --directory heatmaps
"""

import re
from pathlib import Path

# Paths relative to stock_analysis/
SCRIPT_DIR = Path(__file__).resolve().parent
STOCK_ANALYSIS_DIR = SCRIPT_DIR.parent
HEATMAPS_DIR = STOCK_ANALYSIS_DIR / "heatmaps"


def _period_sort_key(name: str) -> tuple:
    if re.match(r"^\d+y$", name):
        return (0, int(name.replace("y", "")))
    if name.isdigit() and len(name) == 4:
        return (1, int(name))
    return (2, name)


def _is_period_folder(name: str) -> bool:
    return bool(re.match(r"^\d+y$", name) or (name.isdigit() and len(name) == 4))


def get_indices() -> list[dict]:
    """If heatmaps/ has index/period structure, return list of index dirs."""
    if not HEATMAPS_DIR.exists():
        return []
    indices = []
    for p in HEATMAPS_DIR.iterdir():
        if not p.is_dir():
            continue
        if any(_is_period_folder(q.name) for q in p.iterdir() if q.is_dir()):
            indices.append({"id": p.name, "label": p.name})
    return sorted(indices, key=lambda x: x["id"])


def get_periods(parent: Path | None = None) -> list[dict]:
    base = parent or HEATMAPS_DIR
    if not base.exists():
        return []
    result = []
    for path in base.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        images = list(path.glob("*.png"))
        comparison = [f for f in images if "performance_comparison" in f.name]
        symbols = [f for f in images if "performance_comparison" not in f.name]
        label = f"Last {name.replace('y', '')} years" if re.match(r"^\d+y$", name) else (f"Year {name}" if name.isdigit() and len(name) == 4 else name)
        result.append({"id": name, "label": label, "comparison_count": len(comparison), "symbol_count": len(symbols)})
    result.sort(key=lambda x: _period_sort_key(x["id"]))
    return result


def get_images(period_id: str, index_id: str | None = None) -> dict:
    if index_id:
        folder = HEATMAPS_DIR / index_id / period_id
    else:
        folder = HEATMAPS_DIR / period_id
    if not folder.exists() or not folder.is_dir():
        return {"comparison": None, "symbols": []}
    all_png = sorted(folder.glob("*.png"))
    comparison = None
    symbols = []
    for p in all_png:
        if "performance_comparison" in p.name:
            if p.name.startswith("0_"):
                comparison = p.name
                break
            if comparison is None:
                comparison = p.name
        else:
            symbols.append(p.name)
    symbols.sort(key=lambda s: s.lower())
    return {"comparison": comparison, "symbols": symbols}


BASE_CSS = """
:root { --bg: #0f0f12; --surface: #18181c; --border: #2a2a30; --text: #e4e4e7; --muted: #71717a; --accent: #22c55e; }
* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; min-height: 100vh; }
.wrap { max-width: 1400px; margin: 0 auto; padding: 1rem 1.5rem; }
header { border-bottom: 1px solid var(--border); padding: 1rem 0; }
h1 { font-size: 1.35rem; margin: 0; }
h1 a { color: inherit; text-decoration: none; }
h2 { font-size: 1.15rem; margin: 0 0 1rem; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem; display: block; color: inherit; margin-bottom: 1rem; max-width: 280px; }
.card:hover { border-color: var(--accent); }
.card h3 { margin: 0 0 0.35rem; }
.card p { margin: 0; font-size: 0.9rem; color: var(--muted); }
.img-wrap { background: var(--bg); border-radius: 8px; overflow: hidden; border: 1px solid var(--border); margin-bottom: 0.5rem; }
img { max-width: 100%; height: auto; display: block; }
.period-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 1rem; }
.symbol-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1.25rem; }
.symbol-img-link { display: block; line-height: 0; }
.symbol-label { font-size: 0.9rem; font-weight: 600; margin: 0 0 0.25rem; }
.symbol-hyperlink { color: #3b82f6; text-decoration: underline; }
.symbol-hyperlink:hover { color: #60a5fa; }
.symbol-yahoo-link { display: inline-block; color: #3b82f6; text-decoration: underline; font-size: 0.85rem; margin: 0; }
.symbol-yahoo-link:hover { color: #60a5fa; }
nav { margin-top: 0.5rem; }
nav a { margin-right: 0.75rem; }
"""


def write_index():
    indices = get_indices()
    if indices:
        links = "".join(
            f'<a href="{i["id"]}/index.html" class="card"><h3>{i["label"]}</h3><p>View periods</p></a>'
            for i in indices
        )
        title = "Select an index"
        msg = "No index folders found. Run returns_heatmap.py to generate heatmaps per index."
    else:
        periods = get_periods()
        links = "".join(
            f'<a href="{p["id"]}/index.html" class="card"><h3>{p["label"]}</h3><p>{p["symbol_count"]} symbol heatmaps</p></a>'
            for p in periods
        )
        title = "Select a period"
        msg = "No heatmap folders found. Run returns_heatmap.py first."
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Financial Markets Analysis</title>
  <style>{BASE_CSS}</style>
</head>
<body>
  <header><div class="wrap"><h1>Financial Markets Analysis</h1></div></header>
  <main><div class="wrap">
    <h2>{title}</h2>
    <div class="period-grid">{links if links else f'<p style="color:var(--muted)">{msg}</p>'}</div>
  </div></main>
</body>
</html>"""
    out = HEATMAPS_DIR / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")
    return indices if indices else None


def write_index_periods(index_id: str):
    parent = HEATMAPS_DIR / index_id
    periods = get_periods(parent)
    links = "".join(
        f'<a href="{p["id"]}/index.html" class="card"><h3>{p["label"]}</h3><p>{p["symbol_count"]} symbol heatmaps</p></a>'
        for p in periods
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{index_id} – Periods</title>
  <style>{BASE_CSS}</style>
</head>
<body>
  <header><div class="wrap"><h1><a href="../index.html">Financial Markets Analysis</a></h1><nav><a href="../index.html">Indexes</a></nav></div></header>
  <main><div class="wrap">
    <h2>{index_id} – Select a period</h2>
    <div class="period-grid">{links if links else '<p style="color:var(--muted)">No periods.</p>'}</div>
  </div></main>
</body>
</html>"""
    out = parent / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")


def write_period(period_id: str, period_label: str, periods: list[dict], index_id: str | None = None):
    images = get_images(period_id, index_id=index_id)
    if index_id:
        back_url = "../index.html"
        nav_links = "".join(
            f'<a href="{p["id"]}/index.html">{"[ " + p["id"] + " ]" if p["id"] == period_id else p["id"]}</a>'
            for p in periods
        )
        nav_links = f'<a href="../../index.html">Indexes</a> <a href="../index.html">{index_id}</a> {nav_links}'
    else:
        back_url = "../index.html"
        nav_links = "".join(
            f'<a href="../{p["id"]}/index.html">{"[ " + p["id"] + " ]" if p["id"] == period_id else p["id"]}</a>'
            for p in periods
        )
    comparison_block = ""
    if images["comparison"]:
        comparison_block = f"""
  <section style="margin-bottom:2rem;">
    <h3 style="font-size:1rem;color:var(--muted);margin-bottom:0.75rem;">Performance comparison · Return (%) · $10K growth</h3>
    <div class="img-wrap"><img src="{images["comparison"]}" alt="Comparison heatmap" loading="lazy"></div>
  </section>"""
    def _symbol_card(name: str) -> str:
        base = name.replace("_monthly_returns_heatmap.png", "").replace(".png", "")
        ticker = base.split("_")[0].upper()
        label = base.replace("_", " ")
        yahoo_url = f"https://finance.yahoo.com/quote/{ticker}" if ticker else "#"
        return f'<div class="symbol-card"><div class="img-wrap"><a href="{yahoo_url}" target="_blank" rel="noopener" class="symbol-img-link" title="View {ticker} on Yahoo Finance"><img src="{name}" alt="{label}" loading="lazy"></a></div><p class="symbol-label"><a href="{yahoo_url}" target="_blank" rel="noopener" class="symbol-hyperlink">{label}</a></p><a href="{yahoo_url}" target="_blank" rel="noopener" class="symbol-yahoo-link">View {ticker} on Yahoo Finance ↗</a></div>'
    symbol_cards = "".join(_symbol_card(name) for name in images["symbols"])
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{period_label} – Financial Markets Analysis</title>
  <style>{BASE_CSS}</style>
</head>
<body>
  <header><div class="wrap"><h1><a href="{('../../index.html' if index_id else '../index.html')}">Financial Markets Analysis</a></h1><nav>{nav_links}</nav></div></header>
  <main><div class="wrap">
    <h2>{period_label}</h2>{comparison_block}
    <h3 style="font-size:1rem;color:var(--muted);margin-bottom:0.75rem;">Monthly returns by symbol</h3>
    <div class="symbol-grid">{symbol_cards if symbol_cards else "<p style='color:var(--muted)'>No images.</p>"}</div>
  </div></main>
</body>
</html>"""
    if index_id:
        out = HEATMAPS_DIR / index_id / period_id / "index.html"
    else:
        out = HEATMAPS_DIR / period_id / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")


def main():
    if not HEATMAPS_DIR.exists():
        print(f"Error: {HEATMAPS_DIR} not found. Run from stock_analysis/ or ensure heatmaps/ exists.")
        return
    indices = write_index()
    if indices:
        for idx in indices:
            index_id = idx["id"]
            write_index_periods(index_id)
            parent = HEATMAPS_DIR / index_id
            periods = get_periods(parent)
            for p in periods:
                write_period(p["id"], p["label"], periods, index_id=index_id)
    else:
        periods = get_periods()
        for p in periods:
            write_period(p["id"], p["label"], periods)
    print("Done. Open heatmaps/index.html in your browser, or run: python -m http.server 8000 --directory heatmaps")


if __name__ == "__main__":
    main()
