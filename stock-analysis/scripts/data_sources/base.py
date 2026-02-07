"""
Base contract for historical price data sources.

Any new source (Alpha Vantage, Polygon, broker API, etc.) should
return a DataFrame with these columns so heatmaps and other code work unchanged.
"""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd  # type: ignore[import-untyped]

# All sources must return a DataFrame with these columns (lowercase)
STANDARD_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


class BaseDataSource(ABC):
    """Abstract base for historical OHLCV data providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name of the source (e.g. 'Yahoo Finance')."""
        ...

    @abstractmethod
    def download(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        Download historical OHLCV data.

        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g. QQQ, AAPL).
        start_date : str
            Start date 'YYYY-MM-DD'.
        end_date : str, optional
            End date 'YYYY-MM-DD'. Default: today.
        interval : str
            Bar interval: '1d', '1wk', '1mo', etc. (support depends on source).

        Returns
        -------
        DataFrame with columns: date, open, high, low, close, volume
        or None if download failed.
        """
        ...

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has standard columns; subset and reorder. Override if needed."""
        if df is None or len(df) == 0:
            return df
        cols = [c for c in STANDARD_COLUMNS if c in df.columns]
        return df[cols].copy()
