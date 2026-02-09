"""
Financial Markets – Web App

Serves the generated heatmap images with navigation by period (2y, 5y, 10y, 20y, etc.).
Run: flask --app app run  (or: python app.py)
Then open http://127.0.0.1:5001
"""

import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from collections import OrderedDict
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError
from wsgiref.handlers import format_date_time

import pandas as pd  # type: ignore[import-untyped]

try:
    import yfinance as yf  # type: ignore[import-untyped]
except ImportError:
    yf = None

try:
    from flask import Flask, Response, render_template, request, send_from_directory, url_for  # type: ignore[import-untyped]
except ModuleNotFoundError:
    print("Flask is required. Use a virtual environment (macOS/Homebrew blocks system-wide pip):")
    print("  cd stock_analysis")
    print("  python3 -m venv .venv")
    print("  source .venv/bin/activate   # on Windows: .venv\\Scripts\\activate")
    print("  pip install flask")
    print("  python app.py")
    print("Or use the run script:  ./run_webapp.sh")
    sys.exit(1)

_APP_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(_APP_DIR / "templates"))
# Heatmaps: use dir that exists and has content (index subdirs). Prefer next to app.py, then cwd.
_heatmaps_next_to_app = (_APP_DIR / "heatmaps").resolve()
_cwd_heatmaps = Path.cwd().resolve() / "heatmaps"

def _pick_heatmaps_dir() -> Path:
    for candidate in (_heatmaps_next_to_app, _cwd_heatmaps):
        if candidate.exists() and candidate.is_dir():
            try:
                if any(candidate.iterdir()):
                    return candidate
            except OSError:
                pass
    return _heatmaps_next_to_app

HEATMAPS_DIR = _pick_heatmaps_dir()
DATA_DIR = _APP_DIR / "data"
SYMBOLS_DIR = _APP_DIR / "symbols"
COMPARE_PERIODS = list(range(1, 21))  # 1Y through 20Y
COMPARE_PERIODS_WITH_YTD: list[str | int] = ["YTD"] + COMPARE_PERIODS  # display order
CALENDAR_WEEKS_AHEAD = 6
DEFAULT_COMPARE_SYMBOLS = "QQQ, SPY, SLV, GLD, SMH, SOXX"  # Nasdaq 100, S&P 500, Silver, Gold, Semis ETFs
DEFAULT_SYMBOL_CHART_SYMBOLS = "QQQ, SPY, SLV, GLD, SMH, SOXX"  # Price charts default


def _period_sort_key(name: str) -> tuple:
    """Sort 2y, 5y, 10y, 20y first, then numeric years (2024), then rest."""
    if re.match(r"^\d+y$", name):
        n = int(name.replace("y", ""))
        return (0, n)
    if name.isdigit() and len(name) == 4:
        return (1, int(name))
    return (2, name)


def _is_period_folder(name: str) -> bool:
    return bool(re.match(r"^\d+y$", name) or (name.isdigit() and len(name) == 4))


def _has_period_subdirs(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(_is_period_folder(p.name) for p in path.iterdir() if p.is_dir())


def get_index_folders() -> list[dict]:
    """If heatmaps/ is organized as index/period/, return list of index folders. Dow30 first (historical heatmaps), then rest alphabetically."""
    if not HEATMAPS_DIR.exists():
        return []
    indices = [
        {"id": p.name, "label": p.name}
        for p in HEATMAPS_DIR.iterdir()
        if p.is_dir() and _has_period_subdirs(p)
    ]
    return sorted(indices, key=lambda x: (0 if x["id"] == "Dow30" else 1, x["id"]))


def get_period_folders(parent: Path | None = None) -> list[dict]:
    """List period folders under parent (or heatmaps/) with display label and image counts."""
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
        label = name
        if re.match(r"^\d+y$", name):
            n = name.replace("y", "")
            label = f"Last {n} year" if n == "1" else f"Last {n} years"
        elif name.isdigit() and len(name) == 4:
            label = f"Year {name}"
        result.append({
            "id": name,
            "label": label,
            "comparison_count": len(comparison),
            "symbol_count": len(symbols),
            "total": len(images),
        })
    result.sort(key=lambda x: _period_sort_key(x["id"]))
    return result


def _filename_to_ticker(filename: str) -> tuple[str, str]:
    """From heatmap filename (e.g. aapl_Technology_20y_monthly_returns_heatmap.png) return (ticker, display_label)."""
    base = filename.replace("_monthly_returns_heatmap.png", "").replace(".png", "")
    parts = base.split("_")
    ticker = (parts[0] or "").upper()
    display = base.replace("_", " ")
    return ticker, display


def get_period_images(period_id: str, index_id: str | None = None) -> dict:
    """Return comparison heatmap path and list of symbol heatmap paths for a period."""
    if index_id:
        folder = HEATMAPS_DIR / index_id / period_id
    else:
        folder = HEATMAPS_DIR / period_id
    if not folder.exists() or not folder.is_dir():
        return {"comparison": None, "symbols": [], "symbol_list": []}
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
    # Build list of {filename, ticker, label, yahoo_url} for template links
    symbol_list = []
    for fn in symbols:
        ticker, display = _filename_to_ticker(fn)
        symbol_list.append({
            "filename": fn,
            "ticker": ticker,
            "label": display,
            "yahoo_url": f"https://finance.yahoo.com/quote/{ticker}" if ticker else "#",
        })
    return {"comparison": comparison, "symbols": symbols, "symbol_list": symbol_list}


def _safe_path(parts: str) -> bool:
    return ".." not in parts and "\0" not in parts


# --- Compare: find data, compute returns for 1Y..20Y ---


def find_symbol_csv(symbol: str) -> Path | None:
    """Find CSV for symbol under data/<index>/ (any index). Returns None if not found."""
    sym = symbol.strip().upper()
    if not sym or not DATA_DIR.exists():
        return None
    name = f"{sym.lower()}_historical_data.csv"
    for sub in DATA_DIR.iterdir():
        if sub.is_dir():
            path = sub / name
            if path.is_file():
                return path
    return None


def load_symbol_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV and parse dates. Use adj_close for returns if present (split/dividend adjusted), else close."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    # Prefer adjusted close for correct returns across splits/dividends
    if "adj_close" in df.columns and df["adj_close"].notna().all():
        df["close"] = df["adj_close"]
    elif "Adj Close" in df.columns and df["Adj Close"].notna().all():
        df["close"] = df["Adj Close"]
    return df


def _last_full_calendar_year(df: pd.DataFrame) -> int:
    """Last calendar year that has data through year-end (or latest year if data is through Dec)."""
    if df.empty or "date" not in df.columns:
        return 0
    max_ts = df["date"].max()
    max_year = int(max_ts.year)
    # If latest data is not near end of year, use previous year as last "full" year for multi-year returns
    year_end = pd.Timestamp(f"{max_year}-12-31", tz="UTC")
    if max_ts < year_end and max_ts < year_end - pd.Timedelta(days=30):
        return max_year - 1
    return max_year


def filter_to_last_n_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    """Filter to full calendar years for the last N years. For N>=2 uses last full calendar year as end (no partial current year)."""
    if df.empty or "date" not in df.columns:
        return df
    end_year = _last_full_calendar_year(df) if years >= 2 else int(df["date"].max().year)
    start_year = end_year - years + 1
    return df[
        (df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)
    ].sort_values("date").reset_index(drop=True)


def total_return_pct(df: pd.DataFrame, years: int) -> float | None:
    """Total return % over the last N full calendar years. None if insufficient data. Uses first close in period and last close in period (split-adjusted when available)."""
    if df.empty or "close" not in df.columns:
        return None
    sub = filter_to_last_n_years(df, years)
    if sub.empty or len(sub) < 2:
        return None
    first = sub["close"].iloc[0]
    last = sub["close"].iloc[-1]
    if first <= 0:
        return None
    return float((last / first - 1.0) * 100.0)


def total_return_ytd(df: pd.DataFrame) -> float | None:
    """Year-to-date return % (Jan 1 of latest year in data through latest date). None if insufficient data."""
    if df.empty or "close" not in df.columns or "date" not in df.columns:
        return None
    year = int(df["date"].max().year)
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    sub = df[df["date"] >= start].reset_index(drop=True)
    if sub.empty or len(sub) < 2:
        return None
    first = sub["close"].iloc[0]
    last = sub["close"].iloc[-1]
    if first <= 0:
        return None
    return float((last / first - 1.0) * 100.0)


def yearly_performance(df: pd.DataFrame) -> list[dict]:
    """For each full calendar year in the data, return year, return %, and $10K end value. Sorted year descending (newest first)."""
    if df.empty or "close" not in df.columns or "date" not in df.columns:
        return []
    df = df.copy()
    df["year"] = df["date"].dt.year
    years = sorted(df["year"].unique())
    out: list[dict] = []
    for year in years:
        sub = df[df["year"] == year].sort_values("date").reset_index(drop=True)
        if len(sub) < 2:
            continue
        first = sub["close"].iloc[0]
        last = sub["close"].iloc[-1]
        if first <= 0:
            continue
        ret_pct = float((last / first - 1.0) * 100.0)
        end_val = 10_000.0 * (1.0 + ret_pct / 100.0)
        out.append({
            "year": year,
            "return_pct": ret_pct,
            "growth_10k": f"${end_val:,.0f}",
        })
    out.sort(key=lambda x: x["year"], reverse=True)
    return out


def cumulative_10k_growth(yearly_rows: list[dict]) -> list[dict]:
    """From year-wise returns (oldest first), compute cumulative $10K value at end of each year."""
    if not yearly_rows:
        return []
    # Sort by year ascending for cumulative math
    rows = sorted(yearly_rows, key=lambda x: x["year"])
    cum: list[dict] = []
    val = 10_000.0
    for r in rows:
        val *= 1.0 + r["return_pct"] / 100.0
        cum.append({
            "year": r["year"],
            "value": val,
            "value_fmt": f"${val:,.0f}",
        })
    return cum


DEFAULT_START_YEAR = 2016


def cumulative_growth_since(df: pd.DataFrame, start_year: int) -> list[dict]:
    """Cumulative $10K growth since start_year: value, individual year return %, and cumulative return % at end of each year."""
    yearly = yearly_performance(df)
    if not yearly:
        return []
    # Restrict to start_year onward, sort ascending for compounding
    rows = sorted([r for r in yearly if r["year"] >= start_year], key=lambda x: x["year"])
    if not rows:
        return []
    cum: list[dict] = []
    val = 10_000.0
    for r in rows:
        val *= 1.0 + r["return_pct"] / 100.0
        cumulative_pct = (val / 10_000.0 - 1.0) * 100.0
        cum.append({
            "year": r["year"],
            "value_fmt": f"${val:,.0f}",
            "return_pct": r["return_pct"],
            "cumulative_pct": cumulative_pct,
        })
    return cum


def build_cumulative_since_table(symbols: list[str], start_year: int) -> dict:
    """For each symbol, cumulative $10K growth since start_year plus per-year return % and cumulative return %. values[year] = {value_fmt, return_pct, cumulative_pct}."""
    all_years: set[int] = set()
    rows: list[dict] = []
    missing: list[str] = []
    for sym in symbols:
        path = find_symbol_csv(sym)
        if path is None:
            missing.append(sym)
            rows.append({
                "symbol": sym,
                "values": {},
                "yahoo_url": f"https://finance.yahoo.com/quote/{sym}",
            })
            continue
        df = load_symbol_data(path)
        cum = cumulative_growth_since(df, start_year)
        # Each entry: value_fmt, return_pct (individual year), cumulative_pct
        values = {r["year"]: {"value_fmt": r["value_fmt"], "return_pct": r["return_pct"], "cumulative_pct": r["cumulative_pct"]} for r in cum}
        for y in values:
            all_years.add(y)
        rows.append({
            "symbol": sym,
            "values": values,
            "yahoo_url": f"https://finance.yahoo.com/quote/{sym}",
        })
    years = sorted(all_years) if all_years else []
    return {"rows": rows, "years": years, "missing": missing, "start_year": start_year}


def performance_for_year(df: pd.DataFrame, year: int) -> tuple[float | None, str | None]:
    """Return (return_pct, growth_10k_fmt) for the given calendar year, or (None, None) if insufficient data."""
    if df.empty or "close" not in df.columns or "date" not in df.columns:
        return None, None
    sub = df[df["date"].dt.year == year].sort_values("date").reset_index(drop=True)
    if len(sub) < 2:
        return None, None
    first = sub["close"].iloc[0]
    last = sub["close"].iloc[-1]
    if first <= 0:
        return None, None
    ret_pct = float((last / first - 1.0) * 100.0)
    growth = f"${10_000.0 * (1.0 + ret_pct / 100.0):,.0f}"
    return ret_pct, growth


def build_year_performance_table(symbols: list[str], year: int) -> dict:
    """For each symbol compute that year's return % and $10K growth. Rows sorted by return descending."""
    rows: list[dict] = []
    missing: list[str] = []
    for sym in symbols:
        path = find_symbol_csv(sym)
        if path is None:
            missing.append(sym)
            rows.append({
                "symbol": sym,
                "return_pct": None,
                "growth_10k": None,
                "yahoo_url": f"https://finance.yahoo.com/quote/{sym}",
            })
            continue
        df = load_symbol_data(path)
        ret_pct, growth = performance_for_year(df, year)
        rows.append({
            "symbol": sym,
            "return_pct": ret_pct,
            "growth_10k": growth,
            "yahoo_url": f"https://finance.yahoo.com/quote/{sym}",
        })
    rows.sort(
        key=lambda r: (r["return_pct"] if r["return_pct"] is not None else float("-inf")),
        reverse=True,
    )
    return {"rows": rows, "missing": missing, "year": year}


MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def monthly_returns_for_year(df: pd.DataFrame, year: int) -> dict | None:
    """For one calendar year: monthly return % (1..12), start-of-month price (1..12), year total. Returns dict with keys 1..12, 'year_pct', 'growth_10k', 'start_prices' or None."""
    if df.empty or "close" not in df.columns or "date" not in df.columns:
        return None
    sub = df[df["date"].dt.year == year].sort_values("date").reset_index(drop=True)
    if len(sub) < 2:
        return None
    df_idx = sub.set_index("date")
    monthly = df_idx["close"].resample("ME").last().dropna()
    if len(monthly) < 2:
        return None
    ret = monthly.pct_change().dropna() * 100.0
    out: dict = {int(month): round(float(ret.iloc[i]), 2) for i, month in enumerate(ret.index.month)}
    for m in range(1, 13):
        out.setdefault(m, None)
    first = float(sub["close"].iloc[0])
    last = float(sub["close"].iloc[-1])
    if first <= 0:
        return None
    year_pct = float((last / first - 1.0) * 100.0)
    out["year_pct"] = year_pct
    out["growth_10k"] = f"${10_000.0 * (1.0 + year_pct / 100.0):,.0f}"
    # Price at start of each month: start of Jan = first close of year; start of month m = end of month m-1
    month_ends = {monthly.index[i].month: float(monthly.iloc[i]) for i in range(len(monthly))}
    start_prices: dict[int, float | None] = {1: round(first, 2)}
    for m in range(2, 13):
        start_prices[m] = round(month_ends[m - 1], 2) if (m - 1) in month_ends else None
    out["start_prices"] = start_prices
    return out


def monthly_returns_since(df: pd.DataFrame, start_year: int) -> list[dict]:
    """From start_year to last full year: for each year, monthly return % (1..12), start-of-month price, and year total. Sorted year descending."""
    if df.empty or "close" not in df.columns or "date" not in df.columns:
        return []
    df_idx = df.set_index("date")
    monthly = df_idx["close"].resample("ME").last().dropna()
    if len(monthly) < 2:
        return []
    monthly = monthly.reset_index()
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month
    ret = monthly["close"].pct_change() * 100.0
    monthly["ret_pct"] = ret
    years = sorted([y for y in monthly["year"].unique() if y >= start_year], reverse=True)
    out: list[dict] = []
    for year in years:
        sub = monthly[monthly["year"] == year].sort_values("month")
        if sub.empty:
            continue
        months_dict = {int(row["month"]): round(float(row["ret_pct"]), 2) for _, row in sub.iterrows() if pd.notna(row["ret_pct"])}
        for m in range(1, 13):
            months_dict.setdefault(m, None)
        year_df = df[df["date"].dt.year == year].sort_values("date")
        if len(year_df) < 2:
            continue
        first = float(year_df["close"].iloc[0])
        last = float(year_df["close"].iloc[-1])
        if first <= 0:
            continue
        year_pct = float((last / first - 1.0) * 100.0)
        year_monthly = year_df.set_index("date")["close"].resample("ME").last().dropna()
        month_ends = {year_monthly.index[i].month: float(year_monthly.iloc[i]) for i in range(len(year_monthly))}
        start_prices: dict[int, float | None] = {1: round(first, 2)}
        for m in range(2, 13):
            start_prices[m] = round(month_ends[m - 1], 2) if (m - 1) in month_ends else None
        out.append({
            "year": year,
            "months": months_dict,
            "start_prices": start_prices,
            "year_pct": year_pct,
            "growth_10k": f"${10_000.0 * (1.0 + year_pct / 100.0):,.0f}",
        })
    return out


def build_monthly_performance_table(
    symbols: list[str], year: int | None = None, since: int | None = None
) -> dict:
    """Build data for monthly performance tab. If year is set: one-year view (symbols x months). If since is set: multi-year per symbol. Prefer year when both set."""
    if not symbols:
        return {"rows": [], "missing": [], "year": None, "since": None, "mode": None}
    mode = "year" if year is not None else ("since" if since is not None else None)
    if mode is None:
        since = since or DEFAULT_START_YEAR
        mode = "since"
    if mode == "year":
        start_year = year
    else:
        start_year = since or DEFAULT_START_YEAR

    rows: list[dict] = []
    missing: list[str] = []
    for sym in symbols:
        path = find_symbol_csv(sym)
        if path is None:
            missing.append(sym)
            rows.append({
                "symbol": sym,
                "yahoo_url": f"https://finance.yahoo.com/quote/{sym}",
                "months": None,
                "years": None,
            })
            continue
        df = load_symbol_data(path)
        if mode == "year":
            months = monthly_returns_for_year(df, year)
            rows.append({
                "symbol": sym,
                "yahoo_url": f"https://finance.yahoo.com/quote/{sym}",
                "months": months,
                "years": None,
            })
        else:
            years_data = monthly_returns_since(df, start_year)
            rows.append({
                "symbol": sym,
                "yahoo_url": f"https://finance.yahoo.com/quote/{sym}",
                "months": None,
                "years": years_data,
            })

    return {
        "rows": rows,
        "missing": missing,
        "year": year if mode == "year" else None,
        "since": start_year if mode == "since" else None,
        "mode": mode,
    }


def monthly_return_stats(df: pd.DataFrame, lookback_years: int = 20) -> tuple[float | None, float | None, int]:
    """Compute mean and std of monthly returns (%). Returns (mean_pct, std_pct, n_months) or (None, None, 0) if insufficient data."""
    if df.empty or "close" not in df.columns or "date" not in df.columns:
        return None, None, 0
    df = filter_to_last_n_years(df, lookback_years)
    if len(df) < 2:
        return None, None, 0
    df_idx = df.set_index("date")
    monthly = df_idx["close"].resample("ME").last().dropna()
    if len(monthly) < 2:
        return None, None, 0
    ret_pct = monthly.pct_change().dropna() * 100.0
    if len(ret_pct) < 2:
        return None, None, 0
    mean_pct = float(ret_pct.mean())
    std_pct = float(ret_pct.std())
    if pd.isna(std_pct) or std_pct <= 0:
        std_pct = 0.0
    return mean_pct, std_pct, len(ret_pct)


def build_price_range_probability(
    symbols: list[str],
    months_ahead: int = 1,
    lookback_years: int = 20,
    next_12_months: bool = False,
) -> dict:
    """Build future monthly price range probability using 1σ, 2σ, 3σ from historical monthly returns. If next_12_months=True, include ranges for each of months 1..12."""
    rows: list[dict] = []
    missing: list[str] = []
    for sym in symbols:
        path = find_symbol_csv(sym)
        if path is None:
            missing.append(sym)
            rows.append({
                "symbol": sym,
                "yahoo_url": f"https://finance.yahoo.com/quote/{sym}",
                "current_price": None,
                "mean_return_pct": None,
                "std_return_pct": None,
                "n_months": 0,
                "ranges": None,
                "monthly_ranges": None,
            })
            continue
        df = load_symbol_data(path)
        if df.empty or len(df) < 2 or "close" not in df.columns:
            missing.append(sym)
            rows.append({
                "symbol": sym,
                "yahoo_url": f"https://finance.yahoo.com/quote/{sym}",
                "current_price": None,
                "mean_return_pct": None,
                "std_return_pct": None,
                "n_months": 0,
                "ranges": None,
                "monthly_ranges": None,
            })
            continue
        current = float(df["close"].iloc[-1])
        if current <= 0:
            missing.append(sym)
            rows.append({
                "symbol": sym,
                "yahoo_url": f"https://finance.yahoo.com/quote/{sym}",
                "current_price": None,
                "mean_return_pct": None,
                "std_return_pct": None,
                "n_months": 0,
                "ranges": None,
                "monthly_ranges": None,
            })
            continue
        mean_pct, std_pct, n_months = monthly_return_stats(df, lookback_years)
        if mean_pct is None or std_pct is None or n_months < 2:
            missing.append(sym)
            rows.append({
                "symbol": sym,
                "yahoo_url": f"https://finance.yahoo.com/quote/{sym}",
                "current_price": round(current, 2),
                "mean_return_pct": None,
                "std_return_pct": None,
                "n_months": n_months,
                "ranges": None,
                "monthly_ranges": None,
            })
            continue

        def price_at_return(current_p: float, r_pct: float) -> float:
            return round(current_p * (1.0 + r_pct / 100.0), 2)

        n = max(1, months_ahead)
        mu_n = mean_pct * n
        sigma_n = std_pct * (n ** 0.5)
        ranges = {
            "low_1sigma": price_at_return(current, mu_n - 1.0 * sigma_n),
            "high_1sigma": price_at_return(current, mu_n + 1.0 * sigma_n),
            "low_2sigma": price_at_return(current, mu_n - 2.0 * sigma_n),
            "high_2sigma": price_at_return(current, mu_n + 2.0 * sigma_n),
            "low_3sigma": price_at_return(current, mu_n - 3.0 * sigma_n),
            "high_3sigma": price_at_return(current, mu_n + 3.0 * sigma_n),
            "expected_price": price_at_return(current, mu_n),
        }

        monthly_ranges: list[dict] | None = None
        if next_12_months:
            monthly_ranges = []
            for m in range(1, 13):
                mu_m = mean_pct * m
                sigma_m = std_pct * (m ** 0.5)
                monthly_ranges.append({
                    "month": m,
                    "expected_price": price_at_return(current, mu_m),
                    "low_1sigma": price_at_return(current, mu_m - 1.0 * sigma_m),
                    "high_1sigma": price_at_return(current, mu_m + 1.0 * sigma_m),
                    "low_2sigma": price_at_return(current, mu_m - 2.0 * sigma_m),
                    "high_2sigma": price_at_return(current, mu_m + 2.0 * sigma_m),
                    "low_3sigma": price_at_return(current, mu_m - 3.0 * sigma_m),
                    "high_3sigma": price_at_return(current, mu_m + 3.0 * sigma_m),
                })

        rows.append({
            "symbol": sym,
            "yahoo_url": f"https://finance.yahoo.com/quote/{sym}",
            "current_price": round(current, 2),
            "mean_return_pct": round(mean_pct, 3),
            "std_return_pct": round(std_pct, 3),
            "n_months": n_months,
            "months_ahead": n,
            "ranges": ranges,
            "monthly_ranges": monthly_ranges,
        })
    return {
        "rows": rows,
        "missing": missing,
        "months_ahead": months_ahead,
        "lookback_years": lookback_years,
        "next_12_months": next_12_months,
    }


def parse_compare_symbols(raw: str) -> list[str]:
    """Parse comma/space separated symbols, uppercase, dedupe order preserved."""
    if not raw or not raw.strip():
        return []
    seen: set[str] = set()
    out: list[str] = []
    for part in re.split(r"[\s,]+", raw.strip()):
        s = part.strip().upper()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def build_comparison_table(symbols: list[str]) -> dict:
    """For each symbol compute return % and $10K growth for YTD, 1Y, 2Y, ... 20Y."""
    rows: list[dict] = []
    missing: list[str] = []
    for sym in symbols:
        path = find_symbol_csv(sym)
        if path is None:
            missing.append(sym)
            rows.append({
                "symbol": sym,
                "returns": {},
                "growth_10k": {},
                "yahoo_url": f"https://finance.yahoo.com/quote/{sym}",
            })
            continue
        df = load_symbol_data(path)
        returns: dict[str | int, float] = {}
        growth_10k: dict[str | int, str] = {}
        pct_ytd = total_return_ytd(df)
        if pct_ytd is not None:
            returns["YTD"] = pct_ytd
            growth_10k["YTD"] = f"${10_000.0 * (1.0 + pct_ytd / 100.0):,.0f}"
        for y in COMPARE_PERIODS:
            pct = total_return_pct(df, y)
            if pct is not None:
                returns[y] = pct
                growth_10k[y] = f"${10_000.0 * (1.0 + pct / 100.0):,.0f}"
        rows.append({
            "symbol": sym,
            "returns": returns,
            "growth_10k": growth_10k,
            "yahoo_url": f"https://finance.yahoo.com/quote/{sym}",
        })
    # Sort by top performer: 20Y return descending (best first); no 20Y data goes to bottom
    rows.sort(
        key=lambda r: r["returns"].get(20, float("-inf")),
        reverse=True,
    )
    return {"rows": rows, "missing": missing, "periods": COMPARE_PERIODS_WITH_YTD}


def _fetch_market_snapshot() -> list[dict]:
    """Fetch major indices (SPY, QQQ, DIA, VIX) current price and % change. Returns list of {symbol, name, price, change_pct, change_label}."""
    if yf is None:
        return []
    symbols = [
        ("SPY", "S&P 500"),
        ("QQQ", "Nasdaq 100"),
        ("DIA", "Dow 30"),
        ("^VIX", "VIX"),
    ]
    out: list[dict] = []
    for sym, name in symbols:
        try:
            t = yf.Ticker(sym)
            price, ch = None, 0.0
            try:
                info = t.fast_info
                if info is not None:
                    price = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
                    prev = getattr(info, "previous_close", None)
                    if price is not None and prev and prev > 0:
                        ch = ((float(price) / float(prev)) - 1.0) * 100.0
            except Exception:
                pass
            if price is None:
                hist = t.history(period="5d")
                if hist is not None and not hist.empty:
                    last = hist["Close"].iloc[-1]
                    prev = hist["Close"].iloc[-2] if len(hist) >= 2 else last
                    price = float(last)
                    ch = ((last / prev) - 1.0) * 100.0 if prev and prev > 0 else 0.0
            if price is not None:
                out.append({
                    "symbol": sym,
                    "name": name,
                    "price": float(price),
                    "change_pct": round(ch, 2),
                    "change_label": "up" if ch >= 0 else "down",
                })
        except Exception:
            continue
    return out


def _fetch_social_sentiment(symbols: list[str] = None) -> list[dict]:
    """Fetch StockTwits sentiment (bullish/bearish %) for given symbols. Free API, no key."""
    import json
    symbols = symbols or ["SPY", "QQQ", "AAPL"]
    out: list[dict] = []
    for i, sym in enumerate(symbols):
        if i > 0 and len(symbols) > 5:
            time.sleep(0.1)  # short delay to avoid hammering StockTwits
        try:
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{sym}.json?limit=100"
            req = Request(url, headers={"User-Agent": "FinancialMarketsAnalysis/1.0"})
            with urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())
            messages = data.get("messages") or []
            bullish = bearish = 0
            for m in messages:
                sent = (m.get("entities") or {}).get("sentiment") or {}
                basic = sent.get("basic") if isinstance(sent, dict) else None
                if basic == "Bullish":
                    bullish += 1
                elif basic == "Bearish":
                    bearish += 1
            total = bullish + bearish
            if total == 0:
                out.append({"symbol": sym, "bullish_pct": 50, "bearish_pct": 50, "message_count": len(messages)})
            else:
                out.append({
                    "symbol": sym,
                    "bullish_pct": round(100.0 * bullish / total, 0),
                    "bearish_pct": round(100.0 * bearish / total, 0),
                    "message_count": len(messages),
                })
        except Exception:
            continue
    return out


def _landing_page_data() -> dict:
    """Data for landing page: market snapshot, social sentiment, top headlines."""
    snapshot = _fetch_market_snapshot()
    sentiment = _fetch_social_sentiment()
    news_data = _fetch_all_news("SPY", rss_limit_per_feed=3, yf_limit=5)
    headlines = (news_data["rss_items"] + news_data["symbol_items"])[:8]
    return {
        "market_snapshot": snapshot,
        "sentiment": sentiment,
        "headlines": headlines,
        "yfinance_available": yf is not None,
    }


@app.route("/")
def landing():
    """Landing page: current market trends, social sentiment, and top news."""
    data = _landing_page_data()
    return render_template("landing.html", **data)


# TTL cache for Index page headlines. 5 min.
_INDEX_PAGE_CACHE: dict[str, tuple[list, float]] = {}
_INDEX_PAGE_CACHE_TTL = 300  # seconds


def _index_page_cached_headlines() -> list[dict]:
    """Return headlines for index page. Uses in-memory cache with 5 min TTL."""
    now = time.time()
    cache_key = "headlines"
    cached = _INDEX_PAGE_CACHE.get(cache_key)
    if cached and (now - cached[1]) < _INDEX_PAGE_CACHE_TTL:
        return cached[0]
    headlines = _fetch_index_headlines(10)
    _INDEX_PAGE_CACHE[cache_key] = (headlines, now)
    return headlines


@app.route("/indices")
def index():
    """Index selector: headlines about indexes, then heatmaps by index. Cached for 5 min."""
    index_folders = get_index_folders()
    headlines = _index_page_cached_headlines()
    return render_template(
        "index_indices.html",
        indices=index_folders,
        headlines=headlines,
    )


@app.route("/index/<index_id>")
def index_periods(index_id: str):
    if not _safe_path(index_id):
        return "Invalid path", 400
    indices = get_index_folders()
    if index_id not in [i["id"] for i in indices]:
        return "Index not found", 404
    parent = HEATMAPS_DIR / index_id
    periods = get_period_folders(parent)
    return render_template("index_periods.html", index_id=index_id, index_label=index_id, periods=periods)


@app.route("/index/<index_id>/<period_id>")
def index_period(index_id: str, period_id: str):
    if not _safe_path(index_id + period_id):
        return "Invalid path", 400
    parent = HEATMAPS_DIR / index_id
    periods = get_period_folders(parent)
    period_ids = [p["id"] for p in periods]
    if period_id not in period_ids:
        return "Period not found", 404
    images = get_period_images(period_id, index_id=index_id)
    label = next((p["label"] for p in periods if p["id"] == period_id), period_id)
    return render_template(
        "period.html",
        period_id=period_id,
        period_label=label,
        comparison=images["comparison"],
        symbols=images["symbols"],
        symbol_list=images.get("symbol_list", []),
        periods=periods,
        index_id=index_id,
        index_label=index_id,
        heatmap_base=f"{index_id}/{period_id}",
    )


# Cache heatmap images in browser/proxy for 7 days (static assets)
HEATMAP_CACHE_MAX_AGE = 7 * 24 * 3600  # 7 days in seconds
# In-memory cache for heatmap bytes (LRU, max items) for fast repeat access
_HEATMAP_BYTES_CACHE: OrderedDict[str, tuple[bytes, float]] = OrderedDict()
_HEATMAP_CACHE_MAX_ITEMS = 200


def _heatmap_path_inside_dir(folder: Path, base: Path) -> bool:
    """True if folder is inside base (Python 3.8 safe: use relative_to, not is_relative_to)."""
    try:
        folder.resolve().relative_to(base.resolve())
        return True
    except (ValueError, OSError, AttributeError):
        return False


def _heatmap_full_path(relative_path: str) -> Path | None:
    """Resolve relative_path (e.g. Dow30/2y/file.png) under HEATMAPS_DIR. Returns Path or None if invalid."""
    relative_path = relative_path.strip("/").replace("\\", "/")
    if not relative_path or ".." in relative_path:
        return None
    full = (HEATMAPS_DIR / relative_path).resolve()
    try:
        if not full.is_file():
            return None
        full.relative_to(HEATMAPS_DIR.resolve())
        return full
    except (ValueError, OSError, AttributeError):
        return None


def _get_heatmap_cached(filepath: str) -> tuple[bytes, float] | None:
    """Return (image_bytes, mtime) for heatmaps/<filepath>, or None. Uses in-memory LRU cache."""
    filepath = filepath.strip("/").replace("\\", "/")
    if not filepath or ".." in filepath:
        return None
    full = _heatmap_full_path(filepath)
    if full is None:
        return None
    cache_key = filepath
    if cache_key in _HEATMAP_BYTES_CACHE:
        _HEATMAP_BYTES_CACHE.move_to_end(cache_key)
        return _HEATMAP_BYTES_CACHE[cache_key]
    try:
        data = full.read_bytes()
        mtime = full.stat().st_mtime
        if len(_HEATMAP_BYTES_CACHE) >= _HEATMAP_CACHE_MAX_ITEMS:
            _HEATMAP_BYTES_CACHE.popitem(last=False)
        _HEATMAP_BYTES_CACHE[cache_key] = (data, mtime)
        return (data, mtime)
    except OSError:
        return None


@app.route("/heatmaps/<path:filepath>")
def heatmap_file(filepath: str):
    """Serve heatmap image from heatmaps/<filepath>. Always read file and return Response (no send_from_directory)."""
    filepath = filepath.strip("/").replace("\\", "/")
    if not _safe_path(filepath):
        return "Invalid path", 400
    parts = [p for p in filepath.split("/") if p]
    if len(parts) < 2:
        return "Not found", 404
    norm_filepath = "/".join(parts)
    # 1) Cache
    cached = _get_heatmap_cached(norm_filepath)
    if cached is not None:
        data, mtime = cached
        resp = Response(data, mimetype="image/png")
        resp.headers["Cache-Control"] = f"public, max-age={HEATMAP_CACHE_MAX_AGE}, immutable"
        resp.headers["Last-Modified"] = format_date_time(int(mtime))
        resp.headers["Content-Length"] = str(len(data))
        return resp
    # 2) Read file directly and return (avoids send_from_directory path issues)
    full = _heatmap_full_path(norm_filepath)
    if full is None:
        return "Not found", 404
    try:
        data = full.read_bytes()
        mtime = full.stat().st_mtime
    except OSError:
        return "Not found", 404
    resp = Response(data, mimetype="image/png")
    resp.headers["Cache-Control"] = f"public, max-age={HEATMAP_CACHE_MAX_AGE}, immutable"
    resp.headers["Last-Modified"] = format_date_time(int(mtime))
    resp.headers["Content-Length"] = str(len(data))
    return resp


@app.route("/heatmaps-debug")
def heatmaps_debug():
    """Debug: show HEATMAPS_DIR path and whether sample files exist. Remove or restrict in production."""
    sample = "Dow30/2y/0_performance_comparison_heatmap_2y.png"
    full = (HEATMAPS_DIR / sample).resolve()
    return (
        f"HEATMAPS_DIR={HEATMAPS_DIR!r}\n"
        f"HEATMAPS_DIR.exists()={HEATMAPS_DIR.exists()}\n"
        f"sample={sample}\n"
        f"full={full!r}\n"
        f"full.is_file()={full.is_file()}\n"
        f"url_for(heatmap_file, filepath={sample!r})={url_for('heatmap_file', filepath=sample)}\n",
        200,
        {"Content-Type": "text/plain; charset=utf-8"},
    )


@app.route("/period/<period_id>")
def period(period_id: str):
    if not _safe_path(period_id):
        return "Invalid period", 400
    periods = get_period_folders()
    period_ids = [p["id"] for p in periods]
    if period_id not in period_ids:
        return "Period not found", 404
    images = get_period_images(period_id)
    label = next((p["label"] for p in periods if p["id"] == period_id), period_id)
    return render_template(
        "period.html",
        period_id=period_id,
        period_label=label,
        comparison=images["comparison"],
        symbols=images["symbols"],
        symbol_list=images.get("symbol_list", []),
        periods=periods,
        index_id=None,
        index_label=None,
        heatmap_base=period_id,
    )


@app.route("/compare")
def compare():
    raw = request.args.get("symbols", "").strip()
    if not raw:
        raw = DEFAULT_COMPARE_SYMBOLS
    symbols = parse_compare_symbols(raw) if raw else []
    result = build_comparison_table(symbols) if symbols else None
    return render_template(
        "compare.html",
        symbols_param=raw,
        symbols_list=symbols,
        result=result,
        periods=COMPARE_PERIODS_WITH_YTD,
    )


CHART_YEARS_DEFAULT = 20


CHART_INDICATORS = ("sma20", "sma50", "sma200", "ema20", "ema50")


def build_chart_series(
    symbols: list[str],
    years: int = 20,
    indicators: list[str] | None = None,
    use_actual_price: bool = False,
) -> dict:
    """Build time series for chart: all series aligned to a common date index. When use_actual_price=False, price normalized to 100 at start; when True, use actual close price (for single-symbol chart). Optional SMA/EMA indicators in same scale."""
    indicators = indicators or []
    raw_series: list[tuple[str, str, list[tuple[str, float | None]]]] = []
    all_dates: set[str] = set()
    missing: list[str] = []
    for sym in symbols:
        path = find_symbol_csv(sym)
        if path is None:
            missing.append(sym)
            continue
        df = load_symbol_data(path)
        df = filter_to_last_n_years(df, years)
        if df.empty or len(df) < 2 or "close" not in df.columns:
            missing.append(sym)
            continue
        close = df["close"].astype(float)
        first = float(close.iloc[0])
        if first <= 0:
            missing.append(sym)
            continue
        date_strs = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10] for d in df["date"]]
        for d in date_strs:
            all_dates.add(d)
        if use_actual_price:
            price_tuples = [(date_strs[i], round(float(close.iloc[i]), 2)) for i in range(len(date_strs))]
        else:
            normalized = (100.0 * close / first).round(2)
            price_tuples = [(date_strs[i], round(float(normalized.iloc[i]), 2)) for i in range(len(date_strs))]
        raw_series.append((sym, "price", price_tuples))
        for ind in indicators:
            if ind == "sma20":
                sma = close.rolling(window=20, min_periods=1).mean()
                vals = (sma if use_actual_price else (100.0 * sma / first)).round(2)
                tuples = [(date_strs[i], None if pd.isna(vals.iloc[i]) else round(float(vals.iloc[i]), 2)) for i in range(len(date_strs))]
                raw_series.append((f"{sym} SMA 20", "sma20", tuples))
            elif ind == "sma50":
                sma = close.rolling(window=50, min_periods=1).mean()
                vals = (sma if use_actual_price else (100.0 * sma / first)).round(2)
                tuples = [(date_strs[i], None if pd.isna(vals.iloc[i]) else round(float(vals.iloc[i]), 2)) for i in range(len(date_strs))]
                raw_series.append((f"{sym} SMA 50", "sma50", tuples))
            elif ind == "sma200":
                sma = close.rolling(window=200, min_periods=1).mean()
                vals = (sma if use_actual_price else (100.0 * sma / first)).round(2)
                tuples = [(date_strs[i], None if pd.isna(vals.iloc[i]) else round(float(vals.iloc[i]), 2)) for i in range(len(date_strs))]
                raw_series.append((f"{sym} SMA 200", "sma200", tuples))
            elif ind == "ema20":
                ema = close.ewm(span=20, adjust=False).mean()
                vals = (ema if use_actual_price else (100.0 * ema / first)).round(2)
                tuples = [(date_strs[i], None if pd.isna(vals.iloc[i]) else round(float(vals.iloc[i]), 2)) for i in range(len(date_strs))]
                raw_series.append((f"{sym} EMA 20", "ema20", tuples))
            elif ind == "ema50":
                ema = close.ewm(span=50, adjust=False).mean()
                vals = (ema if use_actual_price else (100.0 * ema / first)).round(2)
                tuples = [(date_strs[i], None if pd.isna(vals.iloc[i]) else round(float(vals.iloc[i]), 2)) for i in range(len(date_strs))]
                raw_series.append((f"{sym} EMA 50", "ema50", tuples))
    common_dates = sorted(all_dates) if all_dates else []
    series = []
    for sym, typ, tuples in raw_series:
        val_by_date = dict(tuples)
        values = [val_by_date.get(d, None) for d in common_dates]
        series.append({"symbol": sym, "dates": common_dates, "values": values, "type": typ})
    return {
        "series": series,
        "dates": common_dates,
        "missing": missing,
        "years": years,
        "indicators": indicators,
        "use_actual_price": use_actual_price,
    }


@app.route("/charts")
def charts():
    """Performance tab: 20-year price chart (normalized to 100) and optional technical indicators. Data from downloaded CSVs."""
    raw = request.args.get("symbols", "").strip()
    if not raw:
        raw = DEFAULT_COMPARE_SYMBOLS
    symbols = parse_compare_symbols(raw) if raw else []
    years = min(20, max(5, int(request.args.get("years", CHART_YEARS_DEFAULT) or CHART_YEARS_DEFAULT)))
    indicators = [x.strip().lower() for x in request.args.getlist("indicators") if x and x.strip().lower() in CHART_INDICATORS]
    if not indicators and request.args.get("indicators"):
        indicators = [x.strip().lower() for x in request.args.get("indicators", "").split(",") if x.strip().lower() in CHART_INDICATORS]
    chart_data = (
        build_chart_series(symbols[:8], years=years, indicators=indicators)
        if symbols
        else {"series": [], "missing": [], "years": years, "indicators": []}
    )
    chart_data_json = json.dumps(chart_data, default=lambda x: None)
    return render_template(
        "charts.html",
        symbols_param=raw,
        chart_data=chart_data,
        chart_data_json=chart_data_json,
        years=years,
        indicators=indicators,
    )


@app.route("/symbol-chart")
def symbol_chart():
    """Price charts: one chart per symbol (actual price), with optional indicators and total return."""
    raw = request.args.get("symbols", "").strip() or DEFAULT_SYMBOL_CHART_SYMBOLS
    symbols = parse_compare_symbols(raw) if raw else []
    years = min(20, max(5, int(request.args.get("years", CHART_YEARS_DEFAULT) or CHART_YEARS_DEFAULT)))
    indicators = [x.strip().lower() for x in request.args.getlist("indicators") if x and x.strip().lower() in CHART_INDICATORS]
    if not indicators and request.args.get("indicators"):
        indicators = [x.strip().lower() for x in request.args.get("indicators", "").split(",") if x.strip().lower() in CHART_INDICATORS]
    charts: list[dict] = []
    missing: list[str] = []
    for sym in (symbols[:12] if symbols else []):  # cap at 12 charts
        chart_data = build_chart_series([sym], years=years, indicators=indicators, use_actual_price=True)
        if sym in (chart_data.get("missing") or []):
            missing.append(sym)
            continue
        total_return: float | None = None
        if chart_data.get("series") and sym not in (chart_data.get("missing") or []):
            path = find_symbol_csv(sym)
            if path:
                df = load_symbol_data(path)
                total_return = total_return_pct(df, years)
        charts.append({
            "symbol": sym,
            "chart_data": chart_data,
            "chart_data_json": json.dumps(chart_data, default=lambda x: None),
            "total_return_pct": total_return,
            "yahoo_url": f"https://finance.yahoo.com/quote/{sym}",
        })
    return render_template(
        "symbol_chart.html",
        symbols_param=raw,
        charts=charts,
        missing=missing,
        years=years,
        indicators=indicators,
    )


@app.route("/price-range")
def price_range():
    """Future monthly price range probability using 1σ, 2σ, 3σ from historical monthly returns."""
    raw = request.args.get("symbols", "").strip() or DEFAULT_COMPARE_SYMBOLS
    symbols = parse_compare_symbols(raw) if raw else []
    months_ahead = min(12, max(1, int(request.args.get("months_ahead", 1) or 1)))
    lookback_years = min(30, max(5, int(request.args.get("lookback_years", 20) or 20)))
    next_12 = request.args.get("next_12") in ("1", "on", "yes", "true")
    result = (
        build_price_range_probability(
            symbols,
            months_ahead=months_ahead,
            lookback_years=lookback_years,
            next_12_months=next_12,
        )
        if symbols
        else {"rows": [], "missing": [], "months_ahead": 1, "lookback_years": 20, "next_12_months": False}
    )
    return render_template(
        "price_range.html",
        symbols_param=raw,
        result=result,
        months_ahead=months_ahead,
        lookback_years=lookback_years,
        next_12_months=next_12,
    )


# --- Strategies: technical analysis backtests ---

STRATEGY_OPTIONS = [
    {"id": "sma_crossover", "label": "SMA Crossover (50/200)", "desc": "Golden/Death cross. Long when 50d SMA > 200d SMA. Can beat B&H by avoiding bear markets.", "tooltip": "Long when 50-day SMA is above 200-day SMA; otherwise in cash. Signal at close, position next day. Golden cross = go long; death cross = exit.", "buy_sell": "BUY when 50-day SMA crosses above 200-day SMA (golden cross). SELL when 50-day SMA crosses below 200-day SMA (death cross)."},
    {"id": "sma_20_50", "label": "SMA Crossover (20/50)", "desc": "Faster trend. Long when 20d > 50d SMA. More signals; can capture trends earlier.", "tooltip": "Long when 20-day SMA is above 50-day SMA; otherwise in cash. Faster than 50/200; more trades, earlier trend entries.", "buy_sell": "BUY when 20-day SMA crosses above 50-day SMA. SELL when 20-day SMA crosses below 50-day SMA."},
    {"id": "trend_200", "label": "200-day trend", "desc": "Long only when price > 200d SMA. Stay invested in uptrends, exit in downtrends. Often beats B&H on risk-adjusted basis.", "tooltip": "Long when closing price is above 200-day SMA; otherwise in cash. Stay in when above trend, exit when below.", "buy_sell": "BUY when closing price moves above 200-day SMA. SELL when closing price moves below 200-day SMA."},
    {"id": "momentum", "label": "Momentum (20-day)", "desc": "Long when price > 20d SMA. Simple trend-following; can outperform in strong trends.", "tooltip": "Long when closing price is above 20-day SMA; otherwise in cash. Short-term trend following.", "buy_sell": "BUY when closing price moves above 20-day SMA. SELL when closing price moves below 20-day SMA."},
    {"id": "macd", "label": "MACD (12,26,9)", "desc": "Long when MACD line > signal line. Classic trend-following; can beat B&H in trending markets.", "tooltip": "Long when MACD line (12-day EMA − 26-day EMA) is above the 9-day signal line; otherwise in cash.", "buy_sell": "BUY when MACD line crosses above the signal line (9-day EMA of MACD). SELL when MACD line crosses below the signal line."},
    {"id": "rsi", "label": "RSI filter (14)", "desc": "Long when RSI < 70 (avoid overbought). Reduces exposure at overbought.", "tooltip": "Long when RSI(14) is below 70 (avoid overbought); otherwise in cash. Stays out when overbought.", "buy_sell": "BUY when RSI(14) is below 70 (not overbought). SELL when RSI(14) rises to 70 or above (overbought)."},
    {"id": "rsi_oversold", "label": "RSI oversold (14)", "desc": "Long when RSI < 40 (buy dips), out when RSI > 70. Mean reversion; can beat B&H in choppy/range-bound markets.", "tooltip": "Long when RSI(14) drops below 40 (buy dips); exit when RSI rises above 70. Mean-reversion style.", "buy_sell": "BUY when RSI(14) drops below 40 (oversold). SELL when RSI(14) rises above 70 (overbought)."},
]


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(window=n, min_periods=1).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=n, min_periods=n).mean()
    avg_loss = loss.rolling(window=n, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def _backtest_equity(df: pd.DataFrame, position: pd.Series) -> tuple[float, float, int]:
    """Given daily position (0 or 1), compute strategy total return %, buy&hold return %, and number of trades. position aligned to df index."""
    if df.empty or len(df) < 2 or "close" not in df.columns:
        return 0.0, 0.0, 0
    close = df["close"].dropna()
    if len(close) < 2:
        return 0.0, 0.0, 0
    # Buy & hold: (last/first - 1) * 100; avoid cumprod since pct_change()[0] is NaN
    first_close = float(close.iloc[0])
    last_close = float(close.iloc[-1])
    total_bh = ((last_close / first_close) - 1.0) * 100.0 if first_close and first_close > 0 else 0.0
    if pd.isna(total_bh) or not isinstance(total_bh, (int, float)):
        total_bh = 0.0
    pos = position.reindex(df.index).ffill().fillna(0)
    pos = pos.shift(1).fillna(0)  # trade at close, position next day
    ret = df["close"].pct_change()
    strategy_ret = (pos * ret).fillna(0)
    cum_strategy = (1 + strategy_ret).cumprod()
    # First element can be NaN if ret[0] was NaN; use first valid or 1.0
    start_val = 1.0
    for i in range(len(cum_strategy)):
        v = cum_strategy.iloc[i]
        if v > 0 and not pd.isna(v):
            start_val = float(v)
            break
    end_val = float(cum_strategy.iloc[-1]) if not pd.isna(cum_strategy.iloc[-1]) and cum_strategy.iloc[-1] > 0 else 1.0
    total_strategy = ((end_val / start_val) - 1.0) * 100.0 if start_val else 0.0
    if pd.isna(total_strategy) or not isinstance(total_strategy, (int, float)):
        total_strategy = 0.0
    trades = int((pos.diff().abs() > 0.5).sum())
    return total_strategy, total_bh, trades


def _backtest_sma_crossover(df: pd.DataFrame, short: int = 50, long_period: int = 200) -> dict:
    if df.empty or len(df) < long_period:
        return {"strategy_pct": None, "buy_hold_pct": None, "excess_pct": None, "trades": 0, "error": "Insufficient data"}
    df = df.copy()
    df["sma_short"] = _sma(df["close"], short)
    df["sma_long"] = _sma(df["close"], long_period)
    position = (df["sma_short"] > df["sma_long"]).astype(int)
    strat, bh, trades = _backtest_equity(df, position)
    return {"strategy_pct": round(strat, 1), "buy_hold_pct": round(bh, 1), "excess_pct": round(strat - bh, 1), "trades": trades}


def _backtest_rsi(df: pd.DataFrame, period: int = 14, overbought: float = 70.0) -> dict:
    if df.empty or len(df) < period + 5:
        return {"strategy_pct": None, "buy_hold_pct": None, "excess_pct": None, "trades": 0, "error": "Insufficient data"}
    df = df.copy()
    df["rsi"] = _rsi(df["close"], period)
    position = (df["rsi"] < overbought).astype(int)
    strat, bh, trades = _backtest_equity(df, position)
    return {"strategy_pct": round(strat, 1), "buy_hold_pct": round(bh, 1), "excess_pct": round(strat - bh, 1), "trades": trades}


def _backtest_momentum(df: pd.DataFrame, period: int = 20) -> dict:
    if df.empty or len(df) < period + 5:
        return {"strategy_pct": None, "buy_hold_pct": None, "excess_pct": None, "trades": 0, "error": "Insufficient data"}
    df = df.copy()
    df["sma"] = _sma(df["close"], period)
    position = (df["close"] > df["sma"]).astype(int)
    strat, bh, trades = _backtest_equity(df, position)
    return {"strategy_pct": round(strat, 1), "buy_hold_pct": round(bh, 1), "excess_pct": round(strat - bh, 1), "trades": trades}


def _backtest_sma_20_50(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 55:
        return {"strategy_pct": None, "buy_hold_pct": None, "excess_pct": None, "trades": 0, "error": "Insufficient data"}
    df = df.copy()
    df["sma_short"] = _sma(df["close"], 20)
    df["sma_long"] = _sma(df["close"], 50)
    position = (df["sma_short"] > df["sma_long"]).astype(int)
    strat, bh, trades = _backtest_equity(df, position)
    return {"strategy_pct": round(strat, 1), "buy_hold_pct": round(bh, 1), "excess_pct": round(strat - bh, 1), "trades": trades}


def _backtest_trend_200(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 205:
        return {"strategy_pct": None, "buy_hold_pct": None, "excess_pct": None, "trades": 0, "error": "Insufficient data"}
    df = df.copy()
    df["sma200"] = _sma(df["close"], 200)
    position = (df["close"] > df["sma200"]).astype(int)
    strat, bh, trades = _backtest_equity(df, position)
    return {"strategy_pct": round(strat, 1), "buy_hold_pct": round(bh, 1), "excess_pct": round(strat - bh, 1), "trades": trades}


def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def _backtest_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    if df.empty or len(df) < slow + signal + 5:
        return {"strategy_pct": None, "buy_hold_pct": None, "excess_pct": None, "trades": 0, "error": "Insufficient data"}
    df = df.copy()
    df["ema_fast"] = _ema(df["close"], fast)
    df["ema_slow"] = _ema(df["close"], slow)
    df["macd_line"] = df["ema_fast"] - df["ema_slow"]
    df["macd_signal"] = _ema(df["macd_line"], signal)
    position = (df["macd_line"] > df["macd_signal"]).astype(int)
    strat, bh, trades = _backtest_equity(df, position)
    return {"strategy_pct": round(strat, 1), "buy_hold_pct": round(bh, 1), "excess_pct": round(strat - bh, 1), "trades": trades}


def _backtest_rsi_oversold(df: pd.DataFrame, period: int = 14, buy_below: float = 50.0, sell_above: float = 70.0) -> dict:
    """Mean reversion: long when RSI < buy_below (buy dips), out when RSI > sell_above (exit overbought)."""
    if df.empty or len(df) < period + 5:
        return {"strategy_pct": None, "buy_hold_pct": None, "excess_pct": None, "trades": 0, "error": "Insufficient data"}
    df = df.copy()
    df["rsi"] = _rsi(df["close"], period)
    # Long when RSI below midpoint (buy weakness), out when overbought
    position = ((df["rsi"] < sell_above) & (df["rsi"] > 0)).astype(int)
    strat, bh, trades = _backtest_equity(df, position)
    return {"strategy_pct": round(strat, 1), "buy_hold_pct": round(bh, 1), "excess_pct": round(strat - bh, 1), "trades": trades}


def build_strategies_results(symbols: list[str], strategy_id: str, years: int = 10) -> dict:
    """Run backtests. If strategy_id is 'all', run Buy & Hold + all strategies and return comparison matrix. Else single-strategy format."""
    backtest_fns = {
        "sma_crossover": _backtest_sma_crossover,
        "sma_20_50": _backtest_sma_20_50,
        "trend_200": _backtest_trend_200,
        "momentum": _backtest_momentum,
        "macd": _backtest_macd,
        "rsi": _backtest_rsi,
        "rsi_oversold": _backtest_rsi_oversold,
    }
    compare_all = strategy_id == "all"
    rows: list[dict] = []
    missing: list[str] = []
    for sym in symbols:
        path = find_symbol_csv(sym)
        if path is None:
            missing.append(sym)
            if compare_all:
                rows.append({
                    "symbol": sym,
                    "yahoo_url": f"https://finance.yahoo.com/quote/{sym}",
                    "buy_hold_pct": None,
                    "strategies": {sid: {"strategy_pct": None, "excess_pct": None, "trades": 0} for sid in backtest_fns},
                })
            else:
                rows.append({
                    "symbol": sym,
                    "strategy_pct": None,
                    "buy_hold_pct": None,
                    "excess_pct": None,
                    "trades": 0,
                    "yahoo_url": f"https://finance.yahoo.com/quote/{sym}",
                    "error": "No data",
                })
            continue
        df = load_symbol_data(path)
        df = filter_to_last_n_years(df, years)
        if compare_all:
            buy_hold_pct = None
            strategies_out: dict[str, dict] = {}
            for sid, fn in backtest_fns.items():
                res = fn(df)
                bh = res.get("buy_hold_pct")
                if buy_hold_pct is None and bh is not None and not (isinstance(bh, float) and pd.isna(bh)):
                    buy_hold_pct = round(float(bh), 1) if isinstance(bh, (int, float)) else bh
                strat_pct = res.get("strategy_pct")
                exc_pct = res.get("excess_pct")
                if strat_pct is not None and isinstance(strat_pct, float) and pd.isna(strat_pct):
                    strat_pct = None
                if exc_pct is not None and isinstance(exc_pct, float) and pd.isna(exc_pct):
                    exc_pct = None
                strategies_out[sid] = {
                    "strategy_pct": round(float(strat_pct), 1) if strat_pct is not None and not (isinstance(strat_pct, float) and pd.isna(strat_pct)) else None,
                    "excess_pct": round(float(exc_pct), 1) if exc_pct is not None and not (isinstance(exc_pct, float) and pd.isna(exc_pct)) else None,
                    "trades": res.get("trades", 0),
                }
            if buy_hold_pct is not None and isinstance(buy_hold_pct, float) and pd.isna(buy_hold_pct):
                buy_hold_pct = None
            rows.append({
                "symbol": sym,
                "yahoo_url": f"https://finance.yahoo.com/quote/{sym}",
                "buy_hold_pct": buy_hold_pct,
                "strategies": strategies_out,
            })
        else:
            backtest_fn = backtest_fns.get(strategy_id)
            res = backtest_fn(df) if backtest_fn else {"strategy_pct": None, "buy_hold_pct": None, "excess_pct": None, "trades": 0, "error": "Unknown strategy"}
            rows.append({
                "symbol": sym,
                "strategy_pct": res.get("strategy_pct"),
                "buy_hold_pct": res.get("buy_hold_pct"),
                "excess_pct": res.get("excess_pct"),
                "trades": res.get("trades", 0),
                "yahoo_url": f"https://finance.yahoo.com/quote/{sym}",
                "error": res.get("error"),
            })
    strategy_label = "Compare all (Buy & Hold vs strategies)" if compare_all else next(
        (s["label"] for s in STRATEGY_OPTIONS if s["id"] == strategy_id), strategy_id or "—"
    )
    out: dict = {"rows": rows, "missing": missing, "strategy_label": strategy_label, "years": years, "compare_all": compare_all}
    # Find best strategy by average excess vs B&H across symbols (when compare_all)
    if compare_all and rows:
        sid_to_excesses: dict[str, list[float]] = {sid: [] for sid in backtest_fns}
        for row in rows:
            st = row.get("strategies") or {}
            for sid, data in st.items():
                exc = data.get("excess_pct")
                if exc is not None and not (isinstance(exc, float) and pd.isna(exc)):
                    sid_to_excesses.setdefault(sid, []).append(float(exc))
        best_id: str | None = None
        best_avg: float | None = None
        best_beats: int = 0
        for sid, excesses in sid_to_excesses.items():
            if not excesses:
                continue
            avg_excess = sum(excesses) / len(excesses)
            beats = sum(1 for e in excesses if e > 0)
            if best_avg is None or avg_excess > best_avg:
                best_avg = avg_excess
                best_id = sid
                best_beats = beats
        if best_id is not None:
            label = next((s["label"] for s in STRATEGY_OPTIONS if s["id"] == best_id), best_id)
            out["best_strategy_id"] = best_id
            out["best_strategy_label"] = label
            out["best_strategy_avg_excess"] = round(best_avg, 1)
            out["best_strategy_beats_count"] = best_beats
            out["best_strategy_symbol_count"] = len(sid_to_excesses.get(best_id, []))
    return out


@app.route("/strategies")
def strategies():
    raw = request.args.get("symbols", "").strip()
    if not raw:
        raw = DEFAULT_COMPARE_SYMBOLS
    symbols = parse_compare_symbols(raw) if raw else []
    strategy_id = request.args.get("strategy", "all").strip() or "all"
    years = min(20, max(5, int(request.args.get("years", 10) or 10)))
    result = build_strategies_results(symbols, strategy_id, years=years) if symbols else None
    return render_template(
        "strategies.html",
        symbols_param=raw,
        strategies_list=STRATEGY_OPTIONS,
        strategy_id=strategy_id,
        years=years,
        result=result,
    )


# --- News: market news from yfinance and RSS feeds ---

NEWS_RSS_FEEDS = [
    {"name": "Yahoo Finance", "url": "https://finance.yahoo.com/news/rssindex", "ns": False},
    {"name": "Reuters", "url": "https://www.reuters.com/markets/rss/", "ns": False},
    {"name": "CNBC", "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html", "ns": False},
    {"name": "MarketWatch", "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories", "ns": False},
]


def _parse_rss_entry(item: ET.Element, source: str) -> dict | None:
    """Extract title, link, published, summary from RSS item. Returns dict or None."""
    def text(el: ET.Element | None) -> str:
        if el is not None and el.text is not None:
            return (el.text or "").strip()
        return (el.text or "").strip() if el is not None else ""

    # RSS 2.0: item has title, link, pubDate, description
    # Some feeds use dc:date or other namespaces
    title_el = item.find("title")
    link_el = item.find("link")
    if title_el is None or link_el is None:
        return None
    title = text(title_el)
    link = text(link_el)
    if not title or not link:
        return None
    pub = item.find("pubDate")
    if pub is None:
        pub = item.find("{http://purl.org/dc/elements/1.1/}date")
    pub_str = text(pub) if pub is not None else ""
    desc_el = item.find("description")
    summary = text(desc_el) if desc_el is not None else ""
    if summary and len(summary) > 300:
        summary = summary[:297] + "..."
    return {"title": title, "link": link, "published": pub_str, "summary": summary, "source": source}


def _fetch_rss_feed(url: str, source: str, limit: int = 10) -> list[dict]:
    """Fetch RSS feed and return list of {title, link, published, summary, source}. Handles RSS 2.0 and common namespaces."""
    out: list[dict] = []
    try:
        req = Request(url, headers={"User-Agent": "FinancialMarketsAnalysis/1.0"})
        with urlopen(req, timeout=10) as resp:
            raw = resp.read()
    except (URLError, OSError, ValueError):
        return out
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return out
    # RSS 2.0: root is <rss><channel><item>...
    channel = root.find("channel")
    if channel is None:
        channel = root.find("{http://purl.org/rss/1.0/}channel") or root
    items = channel.findall("item") if channel is not None else []
    if not items:
        items = channel.findall("{http://purl.org/rss/1.0/}item") if channel is not None else []
    for item in items[:limit]:
        entry = _parse_rss_entry(item, source)
        if entry:
            out.append(entry)
    return out


def _fetch_yfinance_news(symbol: str, limit: int = 15) -> list[dict]:
    """Fetch news for symbol via yfinance. Returns list of {title, link, published, summary, source}."""
    if yf is None:
        return []
    out: list[dict] = []
    try:
        t = yf.Ticker(symbol)
        news = t.news
        if not news:
            return out
        for n in news[:limit]:
            title = n.get("title") or ""
            link = n.get("link") or n.get("url") or ""
            if not title:
                continue
            pub_ts = n.get("providerPublishTime") or n.get("published")
            if isinstance(pub_ts, (int, float)):
                try:
                    pub_str = datetime.utcfromtimestamp(int(pub_ts)).strftime("%Y-%m-%d %H:%M UTC")
                except Exception:
                    pub_str = ""
            else:
                pub_str = str(pub_ts) if pub_ts else ""
            pub_str = pub_str or ""
            out.append({
                "title": title,
                "link": link,
                "published": pub_str,
                "summary": (n.get("summary") or "")[:300] or "",
                "source": n.get("publisher") or "Yahoo Finance",
            })
    except Exception:
        pass
    return out


def _fetch_all_news(symbol: str | None, rss_limit_per_feed: int = 8, yf_limit: int = 15) -> dict:
    """Fetch market news: RSS feeds (general) + yfinance (for symbol). Returns {rss_items, symbol_items, symbol}."""
    rss_items: list[dict] = []
    for feed in NEWS_RSS_FEEDS:
        rss_items.extend(_fetch_rss_feed(feed["url"], feed["name"], limit=rss_limit_per_feed))
    # Sort by published if we have a parseable date; else keep order
    symbol_items: list[dict] = []
    sym = (symbol or "SPY").strip().upper() or "SPY"
    symbol_items = _fetch_yfinance_news(sym, limit=yf_limit)
    return {"rss_items": rss_items, "symbol_items": symbol_items, "symbol": sym}


def _fetch_index_headlines(limit: int = 10) -> list[dict]:
    """Top headlines about indexes: news for SPY, QQQ, DIA (S&P 500, Nasdaq, Dow). Deduped by link, returns up to limit."""
    index_symbols = ["SPY", "QQQ", "DIA"]  # S&P 500, Nasdaq 100, Dow 30 ETFs
    seen_links: set[str] = set()
    out: list[dict] = []
    for sym in index_symbols:
        items = _fetch_yfinance_news(sym, limit=8)
        for item in items:
            link = (item.get("link") or "").strip()
            if link and link not in seen_links:
                seen_links.add(link)
                out.append(item)
                if len(out) >= limit:
                    return out
        time.sleep(0.1)
    return out[:limit]


@app.route("/news")
def news():
    """Latest market news from RSS feeds and optional symbol-specific news (yfinance)."""
    symbol_param = (request.args.get("symbol") or "SPY").strip().upper() or "SPY"
    data = _fetch_all_news(symbol_param)
    return render_template(
        "news.html",
        symbol_param=symbol_param,
        rss_items=data["rss_items"],
        symbol_items=data["symbol_items"],
        yfinance_available=yf is not None,
    )


# --- Calendar: upcoming earnings by week (NASDAQ100 or given symbol) ---


def load_symbols_for_index(index_name: str) -> list[str]:
    """Load symbol list from symbols/<index_name>/symbols.csv."""
    if not SYMBOLS_DIR.exists():
        return []
    csv_path = SYMBOLS_DIR / index_name / "symbols.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    col = "symbol" if "symbol" in df.columns else df.columns[0]
    symbols = df[col].astype(str).str.strip().str.upper().dropna()
    return [s for s in symbols.tolist() if s and not s.startswith("#")]


def _week_start(d: datetime | pd.Timestamp) -> datetime:
    """Monday of the week (naive datetime for comparison)."""
    if hasattr(d, "to_pydatetime"):
        d = d.to_pydatetime()
    if d.tzinfo:
        d = d.replace(tzinfo=None)
    # Monday = 0
    weekday = d.weekday()
    return (d - timedelta(days=weekday)).replace(hour=0, minute=0, second=0, microsecond=0)


def _earnings_dates_from_dataframe(ed: pd.DataFrame) -> list[pd.Timestamp]:
    """Extract earnings dates from yfinance DataFrame (index or columns)."""
    out: list[pd.Timestamp] = []
    # Index is often DatetimeIndex
    if hasattr(ed.index, "tolist"):
        for idx in ed.index:
            try:
                out.append(pd.Timestamp(idx))
            except Exception:
                pass
    # Columns: try by name then first column
    for col in ed.columns:
        cstr = str(col).lower()
        if "date" in cstr or "earnings" in cstr or "event" in cstr and "start" in cstr:
            for v in ed[col].dropna():
                try:
                    out.append(pd.Timestamp(v))
                except Exception:
                    pass
            if out:
                return out
    if not out and len(ed.columns):
        first = ed.columns[0]
        for v in ed[first].dropna():
            try:
                out.append(pd.Timestamp(v))
            except Exception:
                pass
    return out


def _earnings_from_bulk_calendar(
    symbols: list[str], weeks_ahead: int
) -> dict[datetime, list[dict]]:
    """Try yf.Calendars().get_earnings_calendar() for one bulk request. Returns week_events dict or empty."""
    week_events: dict[datetime, list[dict]] = {}
    if yf is None:
        return week_events
    try:
        start = datetime.utcnow().strftime("%Y-%m-%d")
        end = (datetime.utcnow() + timedelta(weeks=weeks_ahead)).strftime("%Y-%m-%d")
        calendars = yf.Calendars(start=start, end=end)
        df = calendars.get_earnings_calendar(limit=100, filter_most_active=False)
        if df is None or df.empty:
            return week_events
        # DataFrame from get_earnings_calendar: index is Symbol, columns include "Event Start Date" or similar
        symbols_set = {s.upper() for s in symbols}
        date_col = None
        for c in df.columns:
            if "date" in str(c).lower() or ("event" in str(c).lower() and "start" in str(c).lower()):
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]
        for sym, row in df.iterrows():
            sym = str(sym).strip().upper() if sym is not None else ""
            if symbols_set and sym not in symbols_set:
                continue
            try:
                ts = pd.Timestamp(row[date_col])
            except Exception:
                continue
            d = ts.to_pydatetime()
            if d.tzinfo:
                d = d.replace(tzinfo=None)
            now = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            end_cutoff = now + timedelta(weeks=weeks_ahead)
            if d < now or d > end_cutoff:
                continue
            week_start = _week_start(d)
            week_events.setdefault(week_start, [])
            week_events[week_start].append({
                "symbol": sym,
                "date_iso": d.strftime("%Y-%m-%d"),
                "date_label": d.strftime("%b %d"),
            })
    except Exception:
        pass
    return week_events


def get_upcoming_earnings_by_week(
    symbols: list[str], weeks_ahead: int = 6, max_symbols: int | None = 30
) -> list[dict]:
    """Fetch upcoming earnings dates via yfinance, group by week. Returns list of {week_start_iso, week_label, events: [...]}."""
    if not symbols or yf is None:
        return []
    now = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    end_cutoff = now + timedelta(weeks=weeks_ahead)
    week_events: dict[datetime, list[dict]] = {}

    # When many symbols (index mode), try bulk Calendars API first (one request)
    if max_symbols and len(symbols) >= 5:
        week_events = _earnings_from_bulk_calendar(symbols, weeks_ahead)
        if week_events:
            # Build result from bulk; no per-symbol loop
            result = []
            for week_start in sorted(week_events.keys()):
                events = sorted(week_events[week_start], key=lambda x: (x["date_iso"], x["symbol"]))
                result.append({
                    "week_start_iso": week_start.strftime("%Y-%m-%d"),
                    "week_label": week_start.strftime("%b %d, %Y"),
                    "events": events,
                })
            return result

    # Per-symbol fallback (or when user entered specific symbols)
    if max_symbols is not None and len(symbols) > max_symbols:
        symbols = symbols[: max_symbols]
    calendar_delay = 0.45

    for i, sym in enumerate(symbols):
        if i > 0:
            time.sleep(calendar_delay)
        try:
            t = yf.Ticker(sym)
            ed = t.get_earnings_dates(limit=12)
            dates_to_check: list[pd.Timestamp] = []
            if ed is not None and not ed.empty:
                dates_to_check = _earnings_dates_from_dataframe(ed)
            if not dates_to_check:
                try:
                    cal = t.calendar
                    if isinstance(cal, dict) and "Earnings Date" in cal and cal["Earnings Date"]:
                        for v in cal["Earnings Date"]:
                            try:
                                dates_to_check.append(pd.Timestamp(v))
                            except Exception:
                                pass
                except Exception:
                    pass
            for ts in dates_to_check:
                d = ts.to_pydatetime()
                if getattr(d, "tzinfo", None):
                    d = d.replace(tzinfo=None)
                if d < now:
                    continue
                if d > end_cutoff:
                    continue
                week_start = _week_start(d)
                week_events.setdefault(week_start, [])
                date_iso = d.strftime("%Y-%m-%d")
                date_label = d.strftime("%b %d")
                week_events[week_start].append({
                    "symbol": sym,
                    "date_iso": date_iso,
                    "date_label": date_label,
                })
        except Exception:
            continue

    # Sort weeks and build result
    result = []
    for week_start in sorted(week_events.keys()):
        events = sorted(week_events[week_start], key=lambda x: (x["date_iso"], x["symbol"]))
        week_label = week_start.strftime("%b %d, %Y")
        result.append({
            "week_start_iso": week_start.strftime("%Y-%m-%d"),
            "week_label": week_label,
            "events": events,
        })
    return result


def _parse_year(raw: str) -> int | None:
    """Parse year from query param; return None if invalid or empty."""
    s = (raw or "").strip()
    if not s or not s.isdigit():
        return None
    y = int(s)
    if 1990 <= y <= 2100:
        return y
    return None


def _symbol_growth_context(
    symbols_param: str,
    start_year_param: str,
    year_param: str,
    symbols: list[str],
    start_year: int,
    year: int | None,
) -> dict:
    """Build template context for symbol_growth; never returns undefined keys for template."""
    cumulative_since = None
    one_year_result = None
    yahoo_url = None
    error = None

    if not symbols:
        return {
            "symbols_param": symbols_param,
            "start_year_param": str(DEFAULT_START_YEAR),
            "start_year": DEFAULT_START_YEAR,
            "year_param": year_param or "",
            "cumulative_since": None,
            "one_year_result": None,
            "yahoo_url": None,
            "error": None,
        }

    cumulative_since = build_cumulative_since_table(symbols, start_year)
    if cumulative_since["missing"] and len(cumulative_since["rows"]) == 0:
        error = "No data found for any symbol. Download data for an index that includes them."
    elif cumulative_since["missing"] and len(symbols) == 1:
        error = "No data found for this symbol. Download data for an index that includes it."
        yahoo_url = f"https://finance.yahoo.com/quote/{symbols[0]}"
    if year is not None and symbols:
        one_year_result = build_year_performance_table(symbols, year)
        # Merge single-year data into cumulative rows so one table shows symbol + that year + cumulative
        year_by_sym = {r["symbol"]: r for r in one_year_result["rows"]}
        for row in cumulative_since["rows"]:
            sym = row["symbol"]
            yr = year_by_sym.get(sym) or {}
            row["return_pct_yr"] = yr.get("return_pct")
            row["growth_10k_yr"] = yr.get("growth_10k")
    else:
        # Ensure rows have keys so template can use |default safely
        for row in cumulative_since["rows"] if cumulative_since else []:
            row.setdefault("return_pct_yr", None)
            row.setdefault("growth_10k_yr", None)

    return {
        "symbols_param": symbols_param,
        "start_year_param": start_year_param or str(DEFAULT_START_YEAR),
        "start_year": start_year,
        "year_param": year_param or "",
        "cumulative_since": cumulative_since,
        "one_year_result": one_year_result,
        "yahoo_url": yahoo_url,
        "error": error,
    }


@app.route("/symbol-growth")
def symbol_growth():
    """Symbols + start year (default 2016): show cumulative $10K growth since that year. Optional 'year' for single-year comparison table."""
    try:
        symbols_param = request.args.get("symbols", request.args.get("symbol", "") or "").strip()
        if not symbols_param:
            symbols_param = DEFAULT_COMPARE_SYMBOLS
        start_year_param = (request.args.get("start_year") or "").strip()
        year_param = (request.args.get("year") or "").strip()
        symbols = parse_compare_symbols(symbols_param) if symbols_param else []
        start_year = _parse_year(start_year_param) or DEFAULT_START_YEAR
        year = _parse_year(year_param) if year_param else None

        ctx = _symbol_growth_context(
            symbols_param, start_year_param, year_param, symbols, start_year, year
        )
        return render_template("symbol_growth.html", **ctx)
    except Exception as e:
        app.logger.exception("symbol_growth failed")
        return render_template(
            "symbol_growth.html",
            symbols_param=request.args.get("symbols", request.args.get("symbol", "") or "").strip(),
            start_year_param=str(DEFAULT_START_YEAR),
            start_year=DEFAULT_START_YEAR,
            year_param="",
            cumulative_since=None,
            one_year_result=None,
            yahoo_url=None,
            error=f"Something went wrong: {e!s}",
        )


@app.route("/monthly-performance")
def monthly_performance():
    """Monthly returns: 1+ symbols and a year (single year) or since year (multi-year monthly returns)."""
    symbols_param = (request.args.get("symbols") or "").strip() or DEFAULT_COMPARE_SYMBOLS
    symbols = parse_compare_symbols(symbols_param) if symbols_param else []
    year_param = (request.args.get("year") or "").strip()
    since_param = (request.args.get("since") or "").strip()
    year = _parse_year(year_param) if year_param else None
    since = _parse_year(since_param) if since_param else None
    result = (
        build_monthly_performance_table(symbols, year=year, since=since)
        if symbols
        else {"rows": [], "missing": [], "year": None, "since": None, "mode": None}
    )
    since_display = since_param
    if not since_display and result.get("mode") == "since" and result.get("since"):
        since_display = str(result["since"])
    return render_template(
        "monthly_performance.html",
        symbols_param=symbols_param,
        year_param=year_param,
        since_param=since_display,
        result=result,
        month_names=MONTH_NAMES,
    )


@app.route("/calendar")
def calendar():
    """Upcoming earnings calendar: NASDAQ100 and/or a given symbol, grouped by week."""
    index_param = (request.args.get("index") or "NASDAQ100").strip().upper()
    symbol_param = (request.args.get("symbol") or "").strip().upper()
    symbols: list[str] = []
    from_index_only = False
    if symbol_param:
        symbols = [s for s in re.split(r"[\s,]+", symbol_param) if s]
    if not symbols and index_param:
        symbols = load_symbols_for_index(index_param)
        from_index_only = bool(symbols)
    weeks_ahead = min(12, max(1, int(request.args.get("weeks", CALENDAR_WEEKS_AHEAD) or CALENDAR_WEEKS_AHEAD)))
    # Cap symbols when loading full index to avoid Yahoo rate limit; no cap when user entered symbols
    max_sym = 30 if from_index_only else None
    total_index_symbols = len(symbols) if from_index_only else None
    by_week = get_upcoming_earnings_by_week(
        symbols, weeks_ahead=weeks_ahead, max_symbols=max_sym
    ) if symbols else []
    indices = []
    if SYMBOLS_DIR.exists():
        indices = [p.name for p in SYMBOLS_DIR.iterdir() if p.is_dir() and (p / "symbols.csv").exists()]
    return render_template(
        "calendar.html",
        index_param=index_param or "NASDAQ100",
        symbol_param=symbol_param,
        by_week=by_week,
        indices=sorted(indices),
        weeks_ahead=weeks_ahead,
        yfinance_available=yf is not None,
        calendar_symbol_cap=max_sym,
        calendar_symbol_total=total_index_symbols,
    )


if __name__ == "__main__":
    print(f"Heatmaps directory: {HEATMAPS_DIR}")
    print(f"  exists: {HEATMAPS_DIR.exists()}")
    if HEATMAPS_DIR.exists():
        indices = [p.name for p in HEATMAPS_DIR.iterdir() if p.is_dir()]
        print(f"  indices: {indices[:5]}{'...' if len(indices) > 5 else ''}")
    port = int(os.environ.get("PORT", "5001"))
    print(f"  Debug: http://127.0.0.1:{port}/heatmaps-debug")
    # host='0.0.0.0' so tunnels (ngrok, localtunnel) can reach the app from any interface
    app.run(debug=True, host="0.0.0.0", port=port)
