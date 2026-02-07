# Index symbol lists

Each subfolder is named after an index:

- **NASDAQ100** – Nasdaq-100 (~100 symbols)
- **SP500** – S&P 500 (~500 symbols)
- **Dow30** – Dow Jones Industrial Average (30 symbols)
- **Russell2000** – Russell 2000 small-cap (~2000 symbols; CSV may be a subset if full list fetch fails)
- **ETFs** – Popular index-tracking and sector-tracking ETFs (~58 symbols: broad index, SPDR/Vanguard/iShares sectors, thematic, commodities, bonds)

Inside each folder:

- **symbols.csv** – list of ticker symbols for that index (column: `symbol`).

## 1. Update symbol lists (from web)

```bash
cd stock_analysis
python scripts/update_symbols.py
```

Or for one index only:

```bash
python scripts/update_symbols.py --index NASDAQ100
python scripts/update_symbols.py --index SP500
```

## 2. Download data for all symbols in each index

```bash
python scripts/download_by_index.py
```

Options:

- `--index NASDAQ100` or `--index ETFs` – only that index
- `--years 10` – years of history (default 20)
- `--source yahoo` – data source
- `--data /path` – base data directory (default: stock_analysis/data)

Data is saved under **data/<index>/** (e.g. `data/NASDAQ100/aapl_historical_data.csv`).  
Each index in **symbols/** has a matching folder under **data/** (and under **heatmaps/** after step 3).

## 3. Generate heatmaps for an index

After downloading, generate heatmaps for that index’s data:

```bash
# All indices (uses data/<index>/ and writes heatmaps/<index>/<period>/)
python scripts/returns_heatmap.py --years 5

# One index only
python scripts/returns_heatmap.py --index NASDAQ100 --years 5
```

Heatmaps go to **heatmaps/<index>/<period>/** (e.g. `heatmaps/NASDAQ100/5y/`). The web app lists indices, then periods per index.
