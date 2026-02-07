"""Yahoo Finance data source (yfinance)."""

from datetime import datetime
from typing import Optional

import pandas as pd  # type: ignore[import-untyped]
import yfinance as yf  # type: ignore[import-untyped]

from .base import BaseDataSource


class YahooFinanceSource(BaseDataSource):
    """Historical OHLCV data from Yahoo Finance via yfinance."""

    @property
    def name(self) -> str:
        return "Yahoo Finance"

    def download(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"Downloading {symbol} from {self.name} ({start_date} to {end_date})...")
        try:
            ticker = yf.Ticker(symbol)
            # auto_adjust=True (yfinance default): OHLC are split/dividend adjusted so returns are correct
            df = ticker.history(start=start_date, end=end_date, interval=interval, auto_adjust=True)
            if len(df) == 0:
                print(f"⚠️  No data returned for {symbol}")
                return None

            df = df.reset_index()
            df = df.rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })
            df = self.normalize(df)
            print(f"✓ Downloaded {len(df)} rows")
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            return df
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {e}")
            return None
