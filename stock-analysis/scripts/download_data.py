"""
Download Historical Data and Generate Heatmaps

Downloads historical data (default 20 years; use --years N) for one or more symbols,
saves to stock_analysis/data/, and generates heatmaps to stock_analysis/heatmaps/.

Data source is pluggable (default: Yahoo Finance). Add more in stock_analysis/scripts/data_sources/.

Usage (from project root):
  python stock_analysis/scripts/download_data.py                    # 20 years, NASDAQ-100 stocks
  python stock_analysis/scripts/download_data.py --years 10         # 10 years
  python stock_analysis/scripts/download_data.py --years 5 QQQ GLD  # 5 years, specific symbols
  python stock_analysis/scripts/download_data.py --source yahoo SPY

Requirements: pip install -r requirements.txt
"""

import argparse
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta
from urllib.request import Request, urlopen

import pandas as pd  # type: ignore[import-untyped]

# Fallback when all NASDAQ-100 fetches fail: popular ETFs
FALLBACK_SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
    "XLI", "XLB", "XLU", "XLRE", "GLD", "SLV", "TLT", "AGG",
]

# Static NASDAQ-100 list (used when all web sources fail; update periodically)
# Source: Wikipedia Nasdaq-100 "Current components" table
NASDAQ100_STATIC = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMGN", "AMZN",
    "APP", "ARM", "ASML", "AVGO", "AXON", "BKR", "BKNG", "CDNS", "CDW", "CEG",
    "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTSH",
    "DASH", "DDOG", "DXCM", "EA", "EXC", "FANG", "FAST", "FER", "FTNT", "GEHC",
    "GILD", "GOOG", "GOOGL", "HON", "IDXX", "INTC", "INTU", "INSM", "ISRG", "KDP",
    "KLAC", "KHC", "LIN", "LRCX", "MAR", "MCHP", "MELI", "META", "MNST", "MSTR",
    "MU", "MDLZ", "MRVL", "MSFT", "MPWR", "NFLX", "NVDA", "NXPI", "ODFL", "ORLY",
    "PCAR", "PANW", "PAYX", "PEP", "PDD", "PLTR", "PYPL", "QCOM", "REGN", "ROP",
    "ROST", "SBUX", "SHOP", "SNPS", "STX", "TMUS", "TSLA", "TTWO", "TXN", "TRI",
    "VRSK", "VRTX", "WBD", "WDC", "WDAY", "WMT", "XEL", "ZS",
]

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def _fetch_html(url: str) -> str | None:
    """Fetch URL with browser User-Agent; return HTML or None."""
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _fetch_nasdaq100_from_investing() -> list[str] | None:
    """Fetch NASDAQ-100 symbols from Investing.com NQ-100 components page. Returns list or None."""
    url = "https://www.investing.com/indices/nq-100-components"
    html = _fetch_html(url)
    if not html:
        return None
    # Page has tickers in links like [INTC](https://...equities/intel-corp)
    found = re.findall(r"\[([A-Za-z]{2,5})\]\s*\(https://[^)]*equities/", html)
    index_tickers = {"GSPC", "DJI", "IXIC", "NDX", "RUT", "VIX", "NYA", "XAX", "USTEC"}
    symbols = list(dict.fromkeys(t.upper() for t in found if t.upper() not in index_tickers and t.isalpha()))
    if 50 <= len(symbols) <= 120:
        return sorted(symbols)
    return None


def _fetch_nasdaq100_from_yahoo() -> list[str] | None:
    """Fetch NASDAQ-100 symbols from Yahoo Finance NDX components page. Returns list or None."""
    url = "https://finance.yahoo.com/quote/%5ENDX/components/"
    html = _fetch_html(url)
    if not html:
        return None
    for pattern in (
        r'[/"]quote/([A-Za-z]{2,5})[/"]',
        r'finance\.yahoo\.com/quote/([A-Za-z]{2,5})/',
    ):
        found = re.findall(pattern, html)
        if len(found) >= 50:
            break
    else:
        return None
    index_tickers = {"GSPC", "DJI", "IXIC", "NDX", "RUT", "VIX", "NYA", "XAX"}
    symbols = [
        t.upper() for t in dict.fromkeys(found)
        if 2 <= len(t) <= 5 and t.isalpha() and t.upper() not in index_tickers
    ]
    if 50 <= len(symbols) <= 120:
        return sorted(symbols)
    return None


def _fetch_nasdaq100_from_wikipedia() -> list[str] | None:
    """Fetch NASDAQ-100 symbols from Wikipedia Nasdaq-100 page (Current components table). Returns list or None."""
    import io
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        html = _fetch_html(url)
        if html:
            tables = pd.read_html(io.StringIO(html), match="Ticker")  # type: ignore[no-untyped-call]
        else:
            tables = pd.read_html(url, match="Ticker")  # type: ignore[no-untyped-call]
        for tbl in tables:
            # Find Ticker column (flat or MultiIndex)
            col = None
            for c in tbl.columns:
                c0 = c[0] if isinstance(c, tuple) else c
                if str(c0).strip().lower() == "ticker":
                    col = c
                    break
            if col is None:
                continue
            tickers = tbl[col].astype(str).str.strip().str.upper()
            tickers = tickers.drop_duplicates()
            symbols = [t for t in tickers.tolist() if t and len(t) <= 5 and t.isalpha()]
            if len(symbols) >= 50:
                return sorted(symbols)
    except Exception:
        pass
    return None


def get_nasdaq100_symbols() -> list[str]:
    """Fetch current NASDAQ-100 constituent tickers from public sites. Returns sorted list; on failure returns FALLBACK_SYMBOLS."""
    # Try multiple public sources in order
    for fetcher in (
        _fetch_nasdaq100_from_investing,
        _fetch_nasdaq100_from_yahoo,
        _fetch_nasdaq100_from_wikipedia,
    ):
        try:
            symbols = fetcher()
            if symbols:
                return symbols
        except Exception:
            continue
    # Use static list (update periodically) so we still get NDX stocks without web
    print("⚠️  Could not fetch live NASDAQ-100 list; using built-in list.")
    return sorted(NASDAQ100_STATIC)


def calculate_basic_stats(df: pd.DataFrame) -> dict:
    """Calculate basic statistics for the downloaded data"""
    
    stats = {
        'total_days': len(df),
        'start_date': df['date'].min(),
        'end_date': df['date'].max(),
        'start_price': df['close'].iloc[0],
        'end_price': df['close'].iloc[-1],
        'total_return': ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100,
        'avg_volume': df['volume'].mean(),
        'high': df['high'].max(),
        'low': df['low'].min(),
    }
    
    return stats


def print_summary(symbol: str, df: pd.DataFrame):
    """Print a summary of the downloaded data"""
    
    stats = calculate_basic_stats(df)
    
    print("=" * 60)
    print(f"SUMMARY: {symbol}")
    print("=" * 60)
    print(f"Date Range:      {stats['start_date']} to {stats['end_date']}")
    print(f"Trading Days:    {stats['total_days']}")
    print(f"Start Price:     ${stats['start_price']:.2f}")
    print(f"End Price:       ${stats['end_price']:.2f}")
    print(f"Total Return:    {stats['total_return']:.2f}%")
    print(f"Period High:     ${stats['high']:.2f}")
    print(f"Period Low:      ${stats['low']:.2f}")
    print(f"Avg Daily Vol:   {stats['avg_volume']:,.0f}")
    print("=" * 60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download historical data and generate heatmaps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  %(prog)s QQQ GLD\n  %(prog)s --source yahoo SPY\n",
    )
    from data_sources import list_sources
    parser.add_argument(
        "--source", "-s",
        default="yahoo",
        help=f"Data source (default: yahoo). Available: {', '.join(list_sources())}",
    )
    parser.add_argument(
        "--years", "-y",
        type=int,
        default=20,
        metavar="N",
        help="Number of years (e.g. 5, 10, 20). Default: 20.",
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory to save CSV files (default: stock_analysis/data).",
    )
    parser.add_argument(
        "symbols",
        nargs="*",
        help="Ticker symbols. If omitted, fetch and use NASDAQ-100 constituents.",
    )
    args = parser.parse_args()

    if args.symbols:
        SYMBOLS = [s.strip().upper() for s in args.symbols if s.strip()]
    else:
        # No symbols provided → fetch NASDAQ-100 stocks (or fallback to ETFs)
        print("No symbols provided → fetching NASDAQ-100 constituents...")
        SYMBOLS = get_nasdaq100_symbols()
        if SYMBOLS == FALLBACK_SYMBOLS:
            print(f"Using fallback list ({len(SYMBOLS)} ETFs).")
        else:
            print(f"Using {len(SYMBOLS)} NASDAQ-100 symbols.")

    if not SYMBOLS:
        print("No symbols given. Use: python download_data.py QQQ GLD SPY")
        sys.exit(1)

    try:
        from data_sources import get_source
        source = get_source(args.source)
    except Exception as e:
        print(f"Failed to load data source '{args.source}': {e}")
        sys.exit(1)

    root = Path(__file__).resolve().parent.parent
    data_dir = args.data.resolve() if args.data else (root / "data")
    data_dir.mkdir(parents=True, exist_ok=True)

    years = max(1, args.years)
    START_DATE = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    END_DATE = None

    print("=" * 60)
    print("Stock Data Downloader")
    print("=" * 60)
    print(f"\n📡 Source: {source.name}")
    print(f"📅 Date range: {START_DATE} to today ({years} years)")
    print(f"📋 Symbols: {', '.join(SYMBOLS)}")
    print(f"📁 Data folder: {data_dir}\n")

    for i, SYMBOL in enumerate(SYMBOLS):
        print(f"\n📥 Downloading {SYMBOL}...")
        print("-" * 60)

        df = source.download(SYMBOL, START_DATE, END_DATE)

        if df is not None:
            filename = f"{SYMBOL.lower()}_historical_data.csv"
            filepath = data_dir / filename
            df.to_csv(filepath, index=False)
            print(f"\n💾 Data saved to: {filepath}")
            print_summary(SYMBOL, df)
            # Generate heatmaps for this symbol (under heatmaps/20y/ for easy navigation)
            output_dir = data_dir.parent / "heatmaps" / "20y"
            try:
                from returns_heatmap import generate_heatmaps
                print(f"\n📊 Generating heatmaps for {SYMBOL}...")
                generate_heatmaps(SYMBOL, df, output_dir)
            except Exception as e:
                print(f"⚠️  Could not generate heatmaps: {e}")
            # Show head/tail only for first symbol
            if i == 0:
                print("\nFirst 5 rows:")
                print(df.head())
                print("\nLast 5 rows:")
                print(df.tail())
    
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print("Data: stock_analysis/data/  |  Heatmaps: stock_analysis/heatmaps/")
    print("=" * 60)