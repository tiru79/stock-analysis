"""
Data source adapters for historical OHLCV data.

Each source implements the same interface so download_data.py and
downstream code (heatmaps, etc.) work with any provider.

To add a new source:
  1. Add a new file in data_sources/ (e.g. alpha_vantage.py)
  2. Implement a class with download(symbol, start_date, end_date) -> DataFrame
  3. Register it in SOURCES below and in get_source().
"""

from pathlib import Path

from .base import STANDARD_COLUMNS

_PACKAGE_DIR = Path(__file__).resolve().parent


def get_source(name: str = "yahoo"):
    """Get a data source by name. Default: yahoo."""
    name = (name or "yahoo").strip().lower()
    if name == "yahoo":
        from .yahoo_finance import YahooFinanceSource
        return YahooFinanceSource()
    # Add more sources here, e.g.:
    # if name == "alpha_vantage":
    #     from .alpha_vantage import AlphaVantageSource
    #     return AlphaVantageSource()
    raise ValueError(f"Unknown source: {name}. Available: yahoo")


def list_sources():
    """Return list of available source names."""
    return ["yahoo"]
