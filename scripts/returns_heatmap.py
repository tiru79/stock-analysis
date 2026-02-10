"""
Monthly Returns Heatmap (with Year and $10K columns)

Generates one heatmap per symbol: monthly returns Jan–Dec, a Year column (yearly return %),
and a $10K column showing how a hypothetical $10,000 investment grows each year.
Can be run standalone (reads from data/) or called from download_data.py after download.

Standalone usage (default: all indices, all symbols, 1y through 20y):
  python stock_analysis/scripts/returns_heatmap.py                    # all indices, 1y..20y → heatmaps/<index>/1y/ .. 20y/
  python stock_analysis/scripts/returns_heatmap.py --years 5 10 20   # only 5y, 10y, 20y
  python stock_analysis/scripts/returns_heatmap.py --index NASDAQ100   # one index, 1y..20y
  python stock_analysis/scripts/returns_heatmap.py 5 QQQ SPY          # last 5y, symbols QQQ SPY
  python stock_analysis/scripts/returns_heatmap.py --year 2024         # single calendar year
  python stock_analysis/scripts/returns_heatmap.py --data /path/to/csvs

Reads *_historical_data.csv from data dir (default stock_analysis/data), saves to stock_analysis/heatmaps/.
With --index: uses data/<index>/ and heatmaps/<index>/<period>/ to match symbols/<index>/.
Skips generating a heatmap if the output file already exists and is newer than the data file (use --force to re-generate).

Storage: Default DPI=100 and PNG compress_level=6 to keep files smaller. Use --dpi 72 for even smaller files;
use --years 5 10 20 to generate only those periods instead of 1y–20y.
"""

import argparse
import sys
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SYMBOLS_DIR = ROOT / "symbols"
DATA_DIR = ROOT / "data"

# Output image settings (lower DPI = smaller files; 100 is a good balance)
DPI = 100

# ETF/symbol -> sector name (for heatmap titles)
ETF_SECTORS = {
    "SPY": "Broad Market",
    "QQQ": "Broad Market",
    "IWM": "Broad Market",
    "DIA": "Broad Market",
    "VOO": "Broad Market",
    "IVV": "Broad Market",
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "GLD": "Commodities",
    "SLV": "Commodities",
    "TLT": "Bonds",
    "AGG": "Bonds",
    "VGT": "Technology",
    "SOXX": "Technology",
    "SMH": "Technology",
    "AAPL": "Technology",
}

# Fixed order for sector/symbol comparison heatmap (sectors not listed sort last)
SECTOR_ORDER = [
    "Broad Market",
    "Technology",
    "Financials",
    "Healthcare",
    "Consumer Discretionary",
    "Consumer Staples",
    "Industrials",
    "Energy",
    "Materials",
    "Utilities",
    "Real Estate",
    "Commodities",
    "Bonds",
    "Other",
]


def symbol_title(symbol: str) -> str:
    """Return 'SYMBOL (Sector)' or 'SYMBOL' if sector unknown."""
    sector = ETF_SECTORS.get(symbol.upper())
    return f"{symbol} ({sector})" if sector else symbol


def sector_for_filename(symbol: str) -> str:
    """Return sector string safe for filenames (e.g. 'Broad_Market'). Empty if unknown."""
    sector = ETF_SECTORS.get(symbol.upper())
    if not sector:
        return ""
    return sector.replace(" ", "_").replace("/", "_")


def heatmap_filename(
    symbol: str,
    output_dir: Path,
    years: int = 20,
    calendar_year: int | None = None,
    end_year: int | None = None,
) -> Path:
    """Output path for monthly returns heatmap. Include sector and year/period when not 20y."""
    sector_safe = sector_for_filename(symbol)
    base = f"{symbol.lower()}"
    if sector_safe:
        base += f"_{sector_safe}"
    if calendar_year is not None:
        base += f"_{calendar_year}"
    elif years != 20:
        if years == 1 and end_year is not None:
            base += f"_{end_year}"
        elif years != 1:
            base += f"_{years}y"
    base += "_monthly_returns_heatmap.png"
    return output_dir / base


def filter_data_by_timeframe(
    df: pd.DataFrame, years: int = 20, calendar_year: int | None = None
) -> tuple[pd.DataFrame, int | None]:
    """Filter df to full calendar years (Jan–Dec) for last N years, or single calendar year.
    Returns (filtered_df, end_year for 1y filename)."""
    df = df.copy()
    if calendar_year is not None:
        start = pd.Timestamp(f"{calendar_year}-01-01", tz="UTC")
        end = pd.Timestamp(f"{calendar_year}-12-31", tz="UTC")
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        return df, calendar_year
    if years >= 20 or years <= 0:
        return df, None
    # Use full calendar years: Jan 1 – Dec 31 for each year (no rolling window)
    end_year = int(df["date"].max().year)
    start_year = end_year - years + 1
    df = df[(df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)].reset_index(drop=True)
    end_year_for_filename = end_year if years == 1 else None
    return df, end_year_for_filename


def get_data_dir() -> Path:
    """Project data directory (data/)."""
    return ROOT / "data"


def get_output_dir(index_name: str | None = None) -> Path:
    """Project output directory for heatmaps: heatmaps/ or heatmaps/<index_name>/."""
    base = ROOT / "heatmaps"
    if index_name:
        return base / index_name
    return base


def list_indices() -> list[str]:
    """Return index names that have symbols/<index>/symbols.csv."""
    if not SYMBOLS_DIR.exists():
        return []
    return [
        p.name for p in SYMBOLS_DIR.iterdir()
        if p.is_dir() and (p / "symbols.csv").exists()
    ]


def load_symbols_for_index(index_name: str) -> list[str]:
    """Load symbol list from symbols/<index_name>/symbols.csv."""
    csv_path = SYMBOLS_DIR / index_name / "symbols.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    col = "symbol" if "symbol" in df.columns else df.columns[0]
    symbols = df[col].astype(str).str.strip().str.upper().dropna()
    return [s for s in symbols.tolist() if s and not s.startswith("#")]


def parse_years_arg(years_arg: str | int) -> list[int]:
    """Parse a single --years value: an integer (e.g. 5) or a range (e.g. 1-20). Returns list of years."""
    s = str(years_arg).strip()
    if "-" in s:
        parts = s.split("-", 1)
        if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
            lo, hi = int(parts[0].strip()), int(parts[1].strip())
            if 1 <= lo <= hi <= 99:
                return list(range(lo, hi + 1))
    if s.isdigit():
        n = int(s)
        if 1 <= n <= 99:
            return [n]
    return []


def period_folder_name(
    years: int,
    calendar_year: int | None = None,
    end_year: int | None = None,
) -> str:
    """Return subfolder name under heatmaps for easy navigation (e.g. 20y, 5y, 2024)."""
    if calendar_year is not None:
        return str(calendar_year)
    if years == 20:
        return "20y"
    if years == 1 and end_year is not None:
        return str(end_year)
    if years == 1:
        return "1y"
    return f"{years}y"


def find_csv(symbol: str, data_dir: Path | None = None) -> Path:
    """Find CSV for symbol. If data_dir given, look there first; else data/, root, cwd."""
    root = Path(__file__).resolve().parent.parent
    name = f"{symbol.lower()}_historical_data.csv"
    bases = [data_dir] if data_dir is not None and data_dir.exists() else []
    bases += [root / "data", root, Path(".")]
    for base in bases:
        if base is None:
            continue
        path = Path(base) / name
        if path.exists():
            return path.resolve()
    default = data_dir if data_dir is not None else root / "data"
    return (Path(default) / name).resolve()


def symbols_from_data_folder(data_dir: Path | None = None) -> list[str]:
    """Discover all symbols that have historical data in the data folder."""
    data_dir = data_dir or get_data_dir()
    if not data_dir.exists():
        return []
    stem_suffix = "_historical_data"  # path.stem is e.g. qqq_historical_data
    symbols = []
    for path in data_dir.glob("*_historical_data.csv"):
        if path.is_file() and path.stem.lower().endswith(stem_suffix):
            sym = path.stem[: -len(stem_suffix)].upper()
            if sym:
                symbols.append(sym)
    return sorted(symbols)


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV and parse dates. Handles ISO8601-style and mixed formats; invalid rows are dropped."""
    df = pd.read_csv(csv_path)
    # Some CSVs have mixed formats (e.g. "2024-01-15 00:00:00-05:00" or malformed "00:00:00-05:00")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def monthly_returns_matrix(
    df: pd.DataFrame, total_row_label: str = "20Y Total"
) -> pd.DataFrame:
    """Individual monthly returns (Jan–Dec) plus final column = total yearly return for that year."""
    df = df.set_index("date")
    monthly = df["close"].resample("ME").last().dropna()
    ret = monthly.pct_change().dropna() * 100  # percentage
    ret = ret.reset_index()
    ret["year"] = ret["date"].dt.year
    ret["month"] = ret["date"].dt.month

    years = sorted(ret["year"].unique())
    pivot = ret.pivot(index="year", columns="month", values="close")
    pivot = pivot.reindex(index=years, columns=range(1, 13))
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    # Total yearly return: (year-end / year-start) - 1
    start = df["close"].resample("YS").first().dropna()
    end = df["close"].resample("YE").last().dropna()
    yearly_pct = (end.values / start.values - 1) * 100
    year_series = pd.Series(yearly_pct, index=end.index.year)
    pivot["Year"] = pivot.index.map(year_series)

    # Hypothetical $10K growth: cumulative value at end of each year (and total row)
    year_order = sorted(y for y in pivot.index if y != total_row_label)
    if year_order:
        cum = 10_000.0
        tenk_series = {}
        for y in year_order:
            r = year_series.get(y, 0)
            if pd.isna(r):
                r = 0
            cum *= 1 + r / 100
            tenk_series[y] = cum
        pivot["$10K"] = pivot.index.map(lambda i: tenk_series.get(i, np.nan))
    else:
        pivot["$10K"] = np.nan

    # Period total return: (last close / first close) - 1
    total_pct = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    total_row = {c: np.nan for c in pivot.columns}
    total_row["Year"] = total_pct
    total_row["$10K"] = 10_000.0 * (1 + total_pct / 100)
    pivot.loc[total_row_label] = total_row
    return pivot


def plot_monthly_heatmap(
    pivot: pd.DataFrame, symbol: str, output_path: Path, period_label: str = "20Y Total"
) -> None:
    """Draw monthly returns heatmap and save. Most recent year at top, total row at bottom."""
    # Put most recent year at top (reverse year order), keep total row at bottom
    if period_label in pivot.index:
        data_rows = pivot.drop(period_label).sort_index(ascending=False)
        pivot = pd.concat([data_rows, pivot.loc[[period_label]]])
    else:
        pivot = pivot.sort_index(ascending=False)
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(15, max(8, len(pivot) * 0.35)), constrained_layout=True)
    gs = GridSpec(2, 1, height_ratios=[0.14, 0.86], hspace=0.02, figure=fig)
    ax_top = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])
    # Color scale from return columns only (so $10K doesn't dominate)
    return_cols = [c for c in pivot.columns if c != "$10K"]
    plot_data = pivot.copy()
    if "$10K" in pivot.columns:
        plot_data = plot_data.copy()
        plot_data["$10K"] = 0  # neutral color for $10K column
    v = pivot[return_cols].abs().max().max() if return_cols else 60
    v = min(v, 60)  # cap color scale (monthly + yearly column)
    cmap = sns.diverging_palette(10, 130, s=80, l=55, as_cmap=True)
    # Annotations: $10K column as dollar amount, others as return %
    def _annot_cell(val, col: str) -> str:
        if pd.isna(val):
            return ""
        if col == "$10K":
            return f"${val:,.0f}"
        return f"{val:.1f}"
    annot = pd.DataFrame(
        {c: [ _annot_cell(pivot[c].iloc[i], c) for i in range(len(pivot)) ] for c in pivot.columns },
        index=pivot.index,
    )
    sns.heatmap(
        plot_data,
        annot=annot,
        fmt="",
        cmap=cmap,
        center=0,
        vmin=-v,
        vmax=v,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Return (%)"},
    )
    title = symbol_title(symbol)
    ax.set_title(f"{title} · Monthly Returns (%) · Year · $10K growth · Top = most recent year, bottom = {period_label}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_ylabel("")
    # Top subplot (ax_top): year labels only, aligned with heatmap rows
    ax_top.set_ylim(ax.get_ylim())
    ax_top.set_yticks(ax.get_yticks())
    ax_top.set_yticklabels([t.get_text() for t in ax.get_yticklabels()], fontsize=9)
    ax_top.set_ylabel("Year", fontsize=10)
    ax_top.yaxis.set_label_position("left")
    ax_top.yaxis.tick_left()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.xaxis.set_visible(False)
    ax_top.set_xticks([])
    for spine in ("top", "right", "bottom"):
        ax_top.spines[spine].set_visible(False)
    ax_top.patch.set_visible(False)
    plt.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
        pil_kwargs={"compress_level": 6},
    )
    plt.close()
    print(f"Saved: {output_path}")


def yearly_returns_series(df: pd.DataFrame) -> pd.Series:
    """Yearly return % by year: (year-end / year-start) - 1."""
    df = df.set_index("date")
    start = df["close"].resample("YS").first().dropna()
    end = df["close"].resample("YE").last().dropna()
    ret = (end.values / start.values - 1) * 100
    return pd.Series(ret, index=end.index.year)


def build_comparison_matrix(
    symbols: list[str],
    years: int,
    calendar_year: int | None,
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Build matrix: rows = symbol (Sector), columns = years, values = yearly return %. Sorted by sector then symbol."""
    rows = []
    for symbol in symbols:
        csv_path = find_csv(symbol, data_dir)
        if not csv_path.exists():
            continue
        df = load_data(csv_path)
        df, _ = filter_data_by_timeframe(df, years=years, calendar_year=calendar_year)
        if len(df) < 2:
            continue
        ser = yearly_returns_series(df)
        total_pct = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
        row = ser.to_dict()
        row["Total"] = total_pct
        row["$10K"] = 10_000.0 * (1 + total_pct / 100)  # hypothetical $10K growth
        label = symbol_title(symbol)
        rows.append((label, symbol, row))
    if not rows:
        return pd.DataFrame(), ""

    # Sort by performance: Total return descending (best performers first)
    def sort_key(item: tuple) -> tuple:
        _, sym, row_dict = item
        total = row_dict.get("Total", np.nan)
        if pd.isna(total):
            return (1, 0, sym)  # NaN totals last
        return (0, -float(total), sym)  # best return first, then symbol for tiebreak

    rows.sort(key=sort_key)
    # Build DataFrame: index = label, columns = years (sorted) + Total + $10K
    year_cols = sorted(set().union(*(row_dict.keys() for _, _, row_dict in rows)) - {"Total", "$10K"})
    all_cols = year_cols + ["Total", "$10K"]
    data = {col: [row_dict.get(col, np.nan) for _, _, row_dict in rows] for col in all_cols}
    matrix = pd.DataFrame(data, index=[label for label, _, _ in rows], columns=all_cols)
    period_desc = f"year {calendar_year}" if calendar_year else f"last_{years}y" if years != 20 else "20y"
    return matrix, period_desc


def plot_comparison_heatmap(
    matrix: pd.DataFrame, output_path: Path, period_title: str
) -> None:
    """Plot sector/symbol performance comparison heatmap (yearly returns % + $10K growth)."""
    if matrix.empty or len(matrix) == 0:
        return
    # Extra width so $10K column has room for dollar amounts (e.g. $32,516) without overlap
    n_cols = matrix.shape[1]
    has_10k = "$10K" in matrix.columns
    extra_10k = 5.0 if has_10k else 0  # extra inches for $10K column
    fig_width = max(16, n_cols * 0.85 + extra_10k)  # 20y: 22 cols + 5 → ~24 in
    fig, ax = plt.subplots(figsize=(fig_width, max(6, matrix.shape[0] * 0.35)))
    # Color scale from return columns only (Year + Total); $10K column gets neutral color
    return_cols = [c for c in matrix.columns if c != "$10K"]
    plot_data = matrix.copy()
    if has_10k:
        plot_data["$10K"] = 0  # neutral so $10K doesn't dominate colormap
    v = matrix[return_cols].abs().max().max() if return_cols else 80
    v = min(v, 80)
    cmap = sns.diverging_palette(10, 130, s=80, l=55, as_cmap=True)

    def _annot_cell(val, col: str) -> str:
        if pd.isna(val):
            return ""
        if col == "$10K":
            return f"${val:,.0f}"
        return f"{val:.0f}"

    annot = pd.DataFrame(
        {c: [_annot_cell(matrix[c].iloc[i], c) for i in range(len(matrix))] for c in matrix.columns},
        index=matrix.index,
    )
    # Slightly smaller annotations when many columns so $10K values don't overlap
    annot_size = 7 if n_cols > 15 else 9
    sns.heatmap(
        plot_data,
        annot=annot,
        fmt="",
        cmap=cmap,
        center=0,
        vmin=-v,
        vmax=v,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Return (%)"},
        annot_kws={"size": annot_size},
    )
    ax.set_title(f"Sector & symbol performance comparison · {period_title} · Return (%) · $10K growth", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Symbol (Sector)")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
        pil_kwargs={"compress_level": 6},
    )
    plt.close()
    print(f"Saved: {output_path}")


def generate_heatmaps(symbol: str, df: pd.DataFrame, output_dir: Path) -> None:
    """Generate monthly returns heatmap (with Year column) from a dataframe (e.g. after download)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    monthly = monthly_returns_matrix(df)
    plot_monthly_heatmap(monthly, symbol, heatmap_filename(symbol, output_dir))


def main():
    parser = argparse.ArgumentParser(
        description="Generate monthly returns heatmaps (with Year column and period total).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--years", "-y",
        nargs="*",
        default=None,
        metavar="N or N-M",
        help="Periods as 1y, 2y, ... 20y. Default: 1-20 (all). E.g. -y 5 10 20 for subset.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        metavar="YYYY",
        help="Single calendar year (e.g. 2024). Overrides --years.",
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default=None,
        metavar="DIR",
        help="Data directory containing *_historical_data.csv (default: stock_analysis/data).",
    )
    parser.add_argument(
        "--index", "-i",
        nargs="*",
        default=None,
        metavar="INDEX",
        help="Index name(s) to match symbols/ and data/ (e.g. NASDAQ100, ETFs). If omitted, use all indices from symbols/.",
    )
    parser.add_argument(
        "symbols",
        nargs="*",
        help="Optional: year(s) as leading numbers (e.g. 2 5 10) and/or ticker symbols. If omitted, use index symbol list or data folder.",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-generate heatmaps even if output is newer than the data file.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        metavar="N",
        help="DPI for output PNGs (default 100; lower = smaller files, e.g. 72 for web).",
    )
    args = parser.parse_args()

    global DPI
    DPI = max(72, min(200, args.dpi))

    calendar_year = args.year

    # Resolve years list: --years 2 5 10 20, or leading positionals 2 5 10, or default [20]
    rest = list(args.symbols)
    positional_years: list[int] = []
    while rest and rest[0].isdigit() and 1 <= int(rest[0]) <= 99 and calendar_year is None:
        positional_years.append(int(rest.pop(0)))
    positional_symbols = [s.strip().upper() for s in rest if s.strip()]

    if calendar_year is not None:
        years_list = [20]
    elif args.years is not None and len(args.years) > 0:
        expanded: list[int] = []
        for a in args.years:
            expanded.extend(parse_years_arg(a))
        years_list = sorted(set(expanded)) if expanded else list(range(1, 21))
    elif positional_years:
        years_list = sorted(set(positional_years))
    else:
        # Default: all periods 1y, 2y, ... 20y
        years_list = list(range(1, 21))

    # Build list of (index_name, data_dir, symbols, output_base)
    runs: list[tuple[str | None, Path, list[str], Path]] = []
    indices = args.index if args.index else list_indices()

    if indices:
        use_positionals_once = len(indices) == 1 and bool(positional_symbols)
        for index_name in indices:
            data_dir = (args.data.resolve() if (args.data and len(indices) == 1) else DATA_DIR / index_name)
            data_dir = data_dir.resolve()
            if use_positionals_once:
                symbols = positional_symbols
            else:
                symbols = load_symbols_for_index(index_name)
            if not symbols:
                print(f"⚠️  No symbols for {index_name}, skipping.")
                continue
            output_base = get_output_dir(index_name)
            runs.append((index_name, data_dir, symbols, output_base))
    else:
        data_dir = args.data.resolve() if args.data else get_data_dir()
        symbols = positional_symbols or symbols_from_data_folder(data_dir)
        if not symbols:
            print("No historical data found in data folder.")
            print("  Run download_data.py or download_by_index.py first, or pass symbols: python returns_heatmap.py QQQ GLD")
            return
        output_base = get_output_dir(None)
        runs.append((None, data_dir, symbols, output_base))

    print(f"Periods:  {years_list} (generating heatmaps for each)")
    for index_name, data_dir, symbols, _ in runs:
        label = index_name or "default"
        print(f"  {label}: data={data_dir}, {len(symbols)} symbols")

    for index_name, data_dir, symbols, output_base in runs:
        for years in years_list:
            if calendar_year is not None:
                period_desc = f"year {calendar_year}"
            elif years == 20:
                period_desc = "20 years"
            elif years == 1:
                period_desc = "last 1 year"
            else:
                period_desc = f"last {years} years"
            period_folder = period_folder_name(years, calendar_year, None)
            output_dir = output_base / period_folder
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n{'='*60}")
            print(f"{index_name or 'Heatmaps'} · {period_desc}  →  {output_dir}")
            print("=" * 60)

            comp_path = output_dir / f"0_performance_comparison_heatmap_{period_folder}.png"
            max_csv_mtime = max(
                (find_csv(s, data_dir).stat().st_mtime for s in symbols if find_csv(s, data_dir).exists()),
                default=0,
            )
            if not args.force and comp_path.exists() and comp_path.stat().st_mtime >= max_csv_mtime:
                print(f"\n📊 Comparison heatmap... ⏭️ Skipped (up to date)")
            else:
                print(f"\n📊 Generating sector & symbol performance comparison heatmap...")
                comparison_matrix, _ = build_comparison_matrix(symbols, years, calendar_year, data_dir)
                if not comparison_matrix.empty:
                    plot_comparison_heatmap(comparison_matrix, comp_path, period_desc)
                else:
                    print("   No data for comparison heatmap.")

            for symbol in symbols:
                csv_path = find_csv(symbol, data_dir)
                if not csv_path.exists():
                    print(f"⚠️  Skipping {symbol}: CSV not found at {csv_path}")
                    continue
                df = load_data(csv_path)
                df, end_year = filter_data_by_timeframe(df, years=years, calendar_year=calendar_year)
                if len(df) == 0:
                    print(f"⚠️  Skipping {symbol}: No data in timeframe.")
                    continue
                if calendar_year is not None:
                    total_row_label = f"{calendar_year} Total"
                    period_label = total_row_label
                elif years == 20:
                    total_row_label = "20Y Total"
                    period_label = total_row_label
                elif years == 1:
                    total_row_label = "1Y Total"
                    period_label = total_row_label
                else:
                    total_row_label = f"{years}Y Total"
                    period_label = total_row_label
                out_path = heatmap_filename(
                    symbol, output_dir, years=years, calendar_year=calendar_year, end_year=end_year
                )
                if not args.force and out_path.exists() and out_path.stat().st_mtime >= csv_path.stat().st_mtime:
                    print(f"\n  {symbol}... ⏭️ Skipped (heatmap up to date)")
                    continue
                print(f"\n📊 Generating heatmap for {symbol}...")
                monthly = monthly_returns_matrix(df, total_row_label=total_row_label)
                plot_monthly_heatmap(monthly, symbol, out_path, period_label=period_label)

    print(f"\nDone. Heatmaps saved to {ROOT / 'heatmaps'}")


if __name__ == "__main__":
    main()
