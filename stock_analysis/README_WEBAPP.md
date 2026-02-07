# Stock Analysis Heatmaps – Web App

Present your generated heatmaps in a simple web UI.

## Prerequisites

- Heatmaps already generated (e.g. in `heatmaps/2y/`, `heatmaps/5y/`, `heatmaps/20y/`).

## Option 1: Static HTML (no Flask)

No server or extra packages needed. Generate HTML once, then open in a browser:

```bash
cd stock_analysis
python scripts/generate_static_web.py
```

Then either:

- Open **heatmaps/index.html** in your browser (double-click or File → Open), or  
- Serve the folder: `python -m http.server 8000 --directory heatmaps` and open **http://127.0.0.1:8000**

## Option 2: Flask app

On macOS/Homebrew use a virtual environment (run script):

```bash
cd stock_analysis
./run_webapp.sh
```

Or manually create a venv and run:

```bash
cd stock_analysis
python3 -m venv .venv
source .venv/bin/activate
pip install flask
python app.py
```

Then open **http://127.0.0.1:5000**. You can run `python app.py` from any directory; templates and heatmaps are found using the app’s install path.

## What you get

- **Home**: List of periods (2y, 5y, 10y, 20y, or single year) that have heatmaps.
- **Period page**: Performance comparison heatmap at the top, then a grid of per-symbol monthly returns heatmaps. Use the nav to switch between periods.

The app only serves existing PNGs from the `heatmaps/` folder; it does not regenerate them. Run `python scripts/returns_heatmap.py` (or use `download_data.py`) to create or update heatmaps first.
