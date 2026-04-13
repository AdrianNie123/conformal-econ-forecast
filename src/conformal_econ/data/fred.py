"""FRED API ingestion with local parquet caching and validation.

The four series here cover distinct macro regimes and data-generating processes —
inflation persistence, labor market dynamics, output volatility, and commodity
shocks. That heterogeneity is the point: conformal coverage should hold across
all of them, not just the easy ones.

Usage:
    python -m conformal_econ.data.fred            # load all, use cache
    python -m conformal_econ.data.fred --refresh  # force re-fetch from FRED
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from conformal_econ.data.cache import (
    get_cache_path,
    get_sample_path,
    is_stale,
    read_cache,
    write_cache,
)

# All four series from PRD Section 4.1
SERIES: dict[str, dict[str, str]] = {
    "CPIAUCSL": {
        "freq": "M",
        "start": "1960-01-01",
        "name": "CPI Inflation (YoY)",
    },
    "UNRATE": {
        "freq": "M",
        "start": "1960-01-01",
        "name": "Unemployment Rate",
    },
    "A191RL1Q225SBEA": {
        "freq": "Q",
        "start": "1960-01-01",
        "name": "Real GDP Growth",
    },
    "DCOILWTICO": {
        "freq": "M",
        "start": "1986-01-01",
        "name": "WTI Crude Oil Price",
    },
}

# Max fraction of consecutive NaNs before we reject the series
_MAX_CONSEC_NAN_FRAC = 0.05


class DataValidationError(Exception):
    """Raised when a loaded series fails basic sanity checks."""


def load_fred_key() -> str | None:
    """Load FRED API key from environment, returning None if absent.

    Loads .env from the project root before checking os.environ.

    Returns:
        The API key string, or None if not set.
    """
    # Walk up to project root and load .env if it exists
    _env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
    return os.environ.get("FRED_API_KEY")


def validate_series(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """Validate a loaded series against basic economic sanity rules.

    Checks:
    - UNRATE cannot be negative (data corruption signal)
    - No run of consecutive NaNs exceeding 5% of total length

    Args:
        df: DataFrame with a single value column, DatetimeIndex.
        series_id: FRED series identifier, used to apply series-specific checks.

    Returns:
        The same DataFrame if validation passes.

    Raises:
        DataValidationError: If any check fails.
    """
    values = df.iloc[:, 0]

    # Unemployment cannot go negative
    if series_id == "UNRATE" and (values < 0).any():
        raise DataValidationError(
            f"{series_id}: found negative values — likely corrupt data"
        )

    # Check for long NaN runs (> 5% consecutive)
    max_allowed = int(len(df) * _MAX_CONSEC_NAN_FRAC)
    if max_allowed < 1:
        max_allowed = 1

    # Count max consecutive NaN run
    max_run = 0
    current_run = 0
    for v in values:
        if pd.isna(v):
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    if max_run > max_allowed:
        raise DataValidationError(
            f"{series_id}: consecutive NaN run of {max_run} exceeds "
            f"allowed {max_allowed} ({_MAX_CONSEC_NAN_FRAC:.0%} of {len(df)} obs)"
        )

    return df


def _load_from_fred(series_id: str, api_key: str) -> pd.DataFrame:
    """Fetch a single series from FRED and return as a DataFrame.

    Args:
        series_id: FRED series identifier.
        api_key: Valid FRED API key.

    Returns:
        DataFrame with DatetimeIndex and a single column named after series_id.
    """
    # Import here so FRED API is only a hard dependency when actually fetching
    from fredapi import Fred  # type: ignore[import-untyped]

    fred = Fred(api_key=api_key)
    meta = SERIES[series_id]
    raw = fred.get_series(series_id, observation_start=meta["start"])
    df = raw.to_frame(name=series_id)
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "date"
    return df


def _load_sample(series_id: str) -> pd.DataFrame:
    """Load the committed sample parquet for CI/offline use.

    Args:
        series_id: FRED series identifier.

    Returns:
        DataFrame from data/sample/{series_id}.parquet.

    Raises:
        FileNotFoundError: If the sample file doesn't exist.
    """
    path = get_sample_path(series_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No FRED API key and no sample data at {path}. "
            "Run scripts/generate_sample_data.py to create sample files, "
            "or set FRED_API_KEY in .env."
        )
    return pd.read_parquet(path)


def fetch_series(
    series_id: str,
    refresh: bool = False,
    _sample_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch a single FRED series, using cache when fresh enough.

    Priority order:
    1. Local parquet cache (if not stale and refresh=False)
    2. FRED API (if FRED_API_KEY is set)
    3. data/sample/ fallback (for CI / no-key environments)

    Args:
        series_id: FRED series identifier. Must be a key in SERIES.
        refresh: Force re-fetch from FRED even if cache is fresh.
        _sample_dir: Override sample directory (used in tests).

    Returns:
        Validated DataFrame with DatetimeIndex.

    Raises:
        KeyError: If series_id is not in the known SERIES dict.
        DataValidationError: If the loaded data fails validation.
        FileNotFoundError: If no API key and no sample file exists.
    """
    if series_id not in SERIES:
        raise KeyError(f"Unknown series '{series_id}'. Known: {list(SERIES)}")

    cache_path = get_cache_path(series_id)

    # Use cache if it's fresh and refresh not forced
    if not refresh and not is_stale(cache_path):
        cached = read_cache(series_id)
        if cached is not None:
            return validate_series(cached, series_id)

    api_key = load_fred_key()

    if api_key:
        df = _load_from_fred(series_id, api_key)
        write_cache(series_id, df)
    else:
        # No key — fall back to committed sample data
        sample_path = (
            (_sample_dir / f"{series_id}.parquet")
            if _sample_dir is not None
            else get_sample_path(series_id)
        )
        if not sample_path.exists():
            raise FileNotFoundError(
                f"No FRED API key and no sample data at {sample_path}."
            )
        df = pd.read_parquet(sample_path)

    return validate_series(df, series_id)


def load_all_series(
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load all four FRED series, returning a dict keyed by series ID.

    Args:
        refresh: Force re-fetch from FRED for all series.

    Returns:
        Dict mapping series_id to validated DataFrame.
    """
    return {sid: fetch_series(sid, refresh=refresh) for sid in SERIES}


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and cache FRED economic series."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-fetch from FRED, ignoring cache age.",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=list(SERIES.keys()),
        help="Series IDs to fetch (default: all four).",
    )
    args = parser.parse_args()

    for sid in args.series:
        try:
            df = fetch_series(sid, refresh=args.refresh)
            path = get_cache_path(sid)
            print(f"  {sid}: {len(df)} obs → {path}")
        except Exception as e:
            print(f"  {sid}: ERROR — {e}")


if __name__ == "__main__":
    _cli()
