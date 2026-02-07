"""
Update index symbol lists in stock_analysis/symbols/<index>/symbols.csv.

Fetches current constituents from public sources and writes a CSV with a 'symbol' column
for each index (NASDAQ100, SP500, Dow30, Russell2000, ETFs).

Usage (from project root or stock_analysis):
  python stock_analysis/scripts/update_symbols.py
  python stock_analysis/scripts/update_symbols.py --index NASDAQ100
  python stock_analysis/scripts/update_symbols.py --index ETFs
"""

import argparse
import io
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd  # type: ignore[import-untyped]

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SYMBOLS_DIR = ROOT / "symbols"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def _fetch_html(url: str) -> str | None:
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def get_nasdaq100() -> list[str]:
    """Return current NASDAQ-100 symbols (from download_data helper)."""
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from download_data import get_nasdaq100_symbols
        return get_nasdaq100_symbols()
    finally:
        if str(SCRIPT_DIR) in sys.path:
            sys.path.remove(str(SCRIPT_DIR))


def get_sp500() -> list[str]:
    """Fetch S&P 500 symbols from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        html = _fetch_html(url)
        if not html:
            return []
        tables = pd.read_html(io.StringIO(html))  # type: ignore[no-untyped-call]
        df = tables[0]
        col = None
        for c in df.columns:
            if str(c).strip().lower() in ("symbol", "ticker"):
                col = c
                break
        if col is None:
            return []
        symbols = df[col].astype(str).str.strip().str.upper().drop_duplicates().tolist()
        return [s for s in symbols if s and len(s) <= 6 and s.replace(".", "").replace("-", "").isalnum()]
    except Exception as e:
        print(f"⚠️  Could not fetch S&P 500 list: {e}")
        return []


def get_dow30() -> list[str]:
    """Fetch Dow Jones Industrial Average (30) symbols from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
        html = _fetch_html(url)
        if not html:
            return []
        tables = pd.read_html(io.StringIO(html))  # type: ignore[no-untyped-call]
        for df in tables:
            for c in df.columns:
                c0 = c[0] if isinstance(c, tuple) else c
                if str(c0).strip().lower() != "symbol":
                    continue
                symbols = df[c].astype(str).str.strip().str.upper().drop_duplicates().tolist()
                symbols = [s for s in symbols if s and len(s) <= 5 and s.isalpha()]
                if 28 <= len(symbols) <= 32:
                    return sorted(symbols)
    except Exception as e:
        print(f"⚠️  Could not fetch Dow 30 list: {e}")
    return []


def get_russell2000() -> list[str]:
    """Fetch Russell 2000 symbols from Investing.com (small-cap 2000 components). Returns list or empty."""
    url = "https://www.investing.com/indices/smallcap-2000-components"
    html = _fetch_html(url)
    if not html:
        return []
    found = re.findall(r"\[([A-Za-z]{2,5})\]\s*\(https://[^)]*equities/", html)
    index_tickers = {"RUT", "DJI", "GSPC", "IXIC", "VIX"}
    symbols = list(dict.fromkeys(t.upper() for t in found if t.upper() not in index_tickers and t.isalpha()))
    if 500 <= len(symbols) <= 2500:
        return sorted(symbols)
    return []


def get_etfs() -> list[str]:
    """Return curated list of popular index-tracking and sector-tracking ETFs (built-in, no fetch)."""
    # Broad index: S&P 500, Nasdaq, Dow, Russell 2000, total market
    index_etfs = [
        "SPY", "IVV", "VOO", "VTI", "RSP",  # S&P 500 / total US
        "QQQ", "QQQM",  # Nasdaq-100
        "DIA",  # Dow 30
        "IWM", "VTWO",  # Russell 2000
        "VTV", "VUG", "VOE", "VBR",  # value/growth/size
        "VXF",  # extended market (ex-S&P 500)
    ]
    # Sector: SPDRs (XL*), Vanguard (V*), iShares (IY*), popular thematic
    sector_etfs = [
        "XLK", "XLV", "XLE", "XLF", "XLI", "XLY", "XLP", "XLB", "XLU",  # SPDR sectors
        "VGT", "VHT", "VDE", "VFH", "VIS", "VCR", "VDC", "VAW", "VPU",  # Vanguard sectors
        "IYW", "IYH", "IYE", "IYF", "IYJ", "IYC", "IYM", "IDU",  # iShares sectors
        "SMH", "SOXX", "IGV",  # semis / software
        "ARKK", "XBI", "IBB",  # innovation / biotech
        "GLD", "SLV", "USO", "UNG",  # commodities
        "TLT", "IEF", "SHY", "HYG", "LQD",  # bonds
        "VXX", "UVXY",  # vol (for reference)
    ]
    return sorted(set(index_etfs + sector_etfs))


INDEX_FETCHERS = {
    "NASDAQ100": get_nasdaq100,
    "SP500": get_sp500,
    "Dow30": get_dow30,
    "Russell2000": get_russell2000,
    "ETFs": get_etfs,
}


def update_index(index_name: str) -> bool:
    """Fetch symbols for index and write symbols/<index_name>/symbols.csv. Returns True if ok."""
    if index_name not in INDEX_FETCHERS:
        print(f"Unknown index: {index_name}. Available: {', '.join(INDEX_FETCHERS)}")
        return False
    symbols = INDEX_FETCHERS[index_name]()
    if not symbols:
        print(f"No symbols obtained for {index_name}.")
        return False
    out_dir = SYMBOLS_DIR / index_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "symbols.csv"
    pd.DataFrame({"symbol": symbols}).to_csv(out_path, index=False)
    print(f"Wrote {len(symbols)} symbols to {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Update index symbol lists in symbols/<index>/symbols.csv")
    parser.add_argument(
        "--index", "-i",
        default=None,
        help="Update only this index (e.g. NASDAQ100, SP500). If omitted, update all.",
    )
    args = parser.parse_args()
    if not SYMBOLS_DIR.exists():
        SYMBOLS_DIR.mkdir(parents=True)
    indices = [args.index] if args.index else list(INDEX_FETCHERS)
    for name in indices:
        if name not in INDEX_FETCHERS:
            print(f"Skipping unknown index: {name}")
            continue
        update_index(name)


if __name__ == "__main__":
    main()
