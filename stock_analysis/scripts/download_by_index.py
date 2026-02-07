"""
Download historical data for all symbols in each index under symbols/<index>/symbols.csv.

Reads symbol lists from stock_analysis/symbols/<index>/symbols.csv and downloads
OHLCV data for each symbol, saving to data/<index>/<symbol>_historical_data.csv.
Skips downloading if the file already exists and is less than 1 day old (use --force to re-download).

Usage (from project root or stock_analysis):
  python stock_analysis/scripts/download_by_index.py
  python stock_analysis/scripts/download_by_index.py --index NASDAQ100
  python stock_analysis/scripts/download_by_index.py --index SP500 --years 10
  python stock_analysis/scripts/download_by_index.py --force   # re-download all
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SYMBOLS_DIR = ROOT / "symbols"
DATA_DIR = ROOT / "data"


def load_symbols(index_name: str) -> list[str]:
    """Load symbol list from symbols/<index_name>/symbols.csv. Returns list of tickers."""
    csv_path = SYMBOLS_DIR / index_name / "symbols.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    col = "symbol" if "symbol" in df.columns else df.columns[0]
    symbols = df[col].astype(str).str.strip().str.upper().dropna()
    return [s for s in symbols.tolist() if s and not s.startswith("#")]


def list_indices() -> list[str]:
    """Return list of index names (subfolders of symbols/ that contain symbols.csv)."""
    if not SYMBOLS_DIR.exists():
        return []
    return [
        p.name for p in SYMBOLS_DIR.iterdir()
        if p.is_dir() and (p / "symbols.csv").exists()
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Download historical data for all symbols in each index (symbols/<index>/symbols.csv → data/<index>/).",
    )
    parser.add_argument(
        "--index", "-i",
        default=None,
        help="Download only this index (e.g. NASDAQ100, SP500). If omitted, download all indices.",
    )
    parser.add_argument(
        "--years", "-y",
        type=int,
        default=20,
        metavar="N",
        help="Years of history (default: 20).",
    )
    parser.add_argument(
        "--source", "-s",
        default="yahoo",
        help="Data source (default: yahoo).",
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default=None,
        help=f"Base data directory (default: {DATA_DIR}).",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-download even if file exists and is less than 1 day old.",
    )
    args = parser.parse_args()

    base_data = args.data.resolve() if args.data else DATA_DIR
    base_data.mkdir(parents=True, exist_ok=True)

    try:
        from data_sources import get_source
        source = get_source(args.source)
    except Exception as e:
        print(f"Data source error: {e}")
        return 1

    indices = [args.index] if args.index else list_indices()
    if not indices:
        print(f"No index folders found under {SYMBOLS_DIR}. Run update_symbols.py first or create symbols/<index>/symbols.csv.")
        return 1

    years = max(1, args.years)
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    end_date = None

    print("=" * 60)
    print("Download by index")
    print("=" * 60)
    print(f"Source:   {source.name}")
    print(f"Years:    {years} ({start_date} to today)")
    print(f"Indices:  {', '.join(indices)}")
    print(f"Data dir: {base_data}")
    print("=" * 60)

    for index_name in indices:
        symbols = load_symbols(index_name)
        if not symbols:
            print(f"\n⚠️  No symbols in {index_name}, skipping.")
            continue
        out_dir = base_data / index_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 {index_name} ({len(symbols)} symbols) → {out_dir}")
        max_age = timedelta(days=1)
        for i, symbol in enumerate(symbols):
            path = out_dir / f"{symbol.lower()}_historical_data.csv"
            if not args.force and path.exists():
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if datetime.now() - mtime < max_age:
                    print(f"\n  [{i+1}/{len(symbols)}] {symbol}... ⏭️ Skipped (file < 1 day old)")
                    continue
            print(f"\n  [{i+1}/{len(symbols)}] {symbol}...")
            df = source.download(symbol, start_date, end_date)
            if df is not None:
                df.to_csv(path, index=False)
                print(f"  💾 Saved {path}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
