# Data source adapters

All sources return a **standard DataFrame**: columns `date`, `open`, `high`, `low`, `close`, `volume` (lowercase).

## Adding a new source

1. **Create a new module** (e.g. `alpha_vantage.py`) in this folder.

2. **Implement a class** that subclasses `BaseDataSource` from `base.py`:
   - `name` property: display name (e.g. `"Alpha Vantage"`)
   - `download(symbol, start_date, end_date=None, interval="1d")` → `pd.DataFrame | None`  
     Return a DataFrame with the standard columns, or `None` on failure.  
     Use `self.normalize(df)` to trim to standard columns if needed.

3. **Register the source** in `__init__.py`:
   - In `get_source(name)`, add a branch (e.g. `if name == "alpha_vantage": ...`).
   - Add the name to `list_sources()`.

4. **Use it**:  
   `python stock_analysis/scripts/download_data.py --source alpha_vantage QQQ`

## Example skeleton

```python
# data_sources/some_api.py
from .base import BaseDataSource
import pandas as pd

class SomeAPISource(BaseDataSource):
    @property
    def name(self) -> str:
        return "Some API"

    def download(self, symbol, start_date, end_date=None, interval="1d"):
        # Fetch from API, then:
        df = pd.DataFrame(...)  # columns: date, open, high, low, close, volume
        return self.normalize(df)
```
