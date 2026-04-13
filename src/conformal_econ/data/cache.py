"""Local parquet caching layer for FRED series data.

Keeps raw data off the network on repeat runs and makes CI work without
a live API key — tests point at data/sample/ instead.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

# Resolve project root from this file's location:
# src/conformal_econ/data/cache.py -> up 4 levels -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SAMPLE_DIR = DATA_DIR / "sample"


def get_cache_path(series_id: str) -> Path:
    """Return the expected parquet path for a cached series.

    Args:
        series_id: FRED series identifier (e.g. 'CPIAUCSL').

    Returns:
        Path to the parquet file, whether or not it exists yet.
    """
    return CACHE_DIR / f"{series_id}.parquet"


def get_sample_path(series_id: str) -> Path:
    """Return the path to the committed sample parquet for CI fallback.

    Args:
        series_id: FRED series identifier.

    Returns:
        Path to the sample parquet file.
    """
    return SAMPLE_DIR / f"{series_id}.parquet"


def is_stale(path: Path, max_age_days: int = 7) -> bool:
    """Check whether a cached file is older than max_age_days.

    Args:
        path: Path to the cached file.
        max_age_days: Maximum age in days before the cache is considered stale.

    Returns:
        True if the file doesn't exist or is older than max_age_days.
    """
    if not path.exists():
        return True
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds > max_age_days * 86_400


def read_cache(series_id: str) -> pd.DataFrame | None:
    """Read a cached series from parquet, returning None if not found.

    Args:
        series_id: FRED series identifier.

    Returns:
        DataFrame with DatetimeIndex, or None if the cache file doesn't exist.
    """
    path = get_cache_path(series_id)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def write_cache(series_id: str, df: pd.DataFrame) -> None:
    """Persist a series DataFrame to parquet cache.

    Args:
        series_id: FRED series identifier.
        df: DataFrame to cache. Index should be a DatetimeIndex.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(get_cache_path(series_id))
