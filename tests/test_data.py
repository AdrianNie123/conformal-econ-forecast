"""Tests for the data pipeline: cache, FRED ingestion, and regime labeling.

All tests use synthetic data. No real FRED API calls.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from conformal_econ.data.cache import (
    is_stale,
    read_cache,
    write_cache,
)
from conformal_econ.data.fred import DataValidationError, fetch_series, validate_series
from conformal_econ.data.regime import label_regimes

# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


def test_cache_write_read(cache_dir: Path) -> None:
    """Written parquet should round-trip back to the same DataFrame."""
    idx = pd.date_range("2000-01-01", periods=10, freq="MS")
    df = pd.DataFrame({"TESTID": np.arange(10, dtype=float)}, index=idx)
    df.index.name = "date"

    # Patch CACHE_DIR so we write into the temp dir
    with patch("conformal_econ.data.cache.CACHE_DIR", cache_dir):
        write_cache("TESTID", df)
        result = read_cache("TESTID")

    assert result is not None
    # Parquet doesn't preserve DatetimeIndex freq metadata — compare values only
    pd.testing.assert_frame_equal(df, result, check_freq=False)


def test_cache_staleness_fresh(cache_dir: Path) -> None:
    """A file written just now should not be stale with default 7-day window."""
    idx = pd.date_range("2000-01-01", periods=5, freq="MS")
    df = pd.DataFrame({"FRESH": np.ones(5)}, index=idx)

    with patch("conformal_econ.data.cache.CACHE_DIR", cache_dir):
        write_cache("FRESH", df)
        path = cache_dir / "FRESH.parquet"

    assert not is_stale(path, max_age_days=7)


def test_cache_staleness_old(cache_dir: Path, tmp_path: Path) -> None:
    """A file with an old mtime should be stale."""
    old_file = tmp_path / "OLD.parquet"
    old_file.write_bytes(b"x")
    # Set mtime to 8 days ago
    old_mtime = time.time() - 8 * 86_400
    import os

    os.utime(old_file, (old_mtime, old_mtime))

    assert is_stale(old_file, max_age_days=7)


def test_cache_miss_returns_none(cache_dir: Path) -> None:
    """read_cache should return None when no file exists."""
    with patch("conformal_econ.data.cache.CACHE_DIR", cache_dir):
        result = read_cache("NONEXISTENT_SERIES")
    assert result is None


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_validate_series_rejects_negative_unemployment() -> None:
    """UNRATE with a negative value should raise DataValidationError."""
    idx = pd.date_range("2000-01-01", periods=10, freq="MS")
    df = pd.DataFrame({"UNRATE": [5.0, 4.5, -0.1, 4.0, 3.9, 4.1, 4.3, 4.5, 4.6, 4.7]}, index=idx)

    with pytest.raises(DataValidationError, match="negative"):
        validate_series(df, "UNRATE")


def test_validate_series_rejects_excess_nans() -> None:
    """A series with a long NaN run (>5% consecutive) should raise DataValidationError."""
    n = 100
    values = [1.0] * n
    # Insert a run of 8 consecutive NaNs starting at index 10 (8% of 100)
    for i in range(10, 18):
        values[i] = float("nan")
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    df = pd.DataFrame({"CPIAUCSL": values}, index=idx)

    with pytest.raises(DataValidationError, match="consecutive NaN"):
        validate_series(df, "CPIAUCSL")


def test_validate_series_passes_clean_data() -> None:
    """A clean series should pass validation and return the same DataFrame."""
    idx = pd.date_range("2000-01-01", periods=20, freq="MS")
    df = pd.DataFrame({"UNRATE": np.linspace(4.0, 6.0, 20)}, index=idx)
    result = validate_series(df, "UNRATE")
    pd.testing.assert_frame_equal(df, result)


# ---------------------------------------------------------------------------
# Regime labeling tests
# ---------------------------------------------------------------------------


def test_regime_labeling_recession_flag() -> None:
    """Dates within a known NBER recession window should be labeled 'recession'."""
    # 2009-01 is well within the 2007-12 to 2009-06 GFC recession
    idx = pd.date_range("2009-01-01", periods=3, freq="MS")
    df = pd.DataFrame({"value": [5.0, 5.5, 6.0]}, index=idx)
    result = label_regimes(df, "UNRATE")
    assert (result["regime"] == "recession").all(), (
        f"Expected all recession, got: {result['regime'].tolist()}"
    )


def test_regime_labeling_expansion_flag() -> None:
    """Dates in a clear expansion period should not be 'recession'."""
    # Mid-2005: between 2001 and 2007 recessions
    # Pad with history so rolling std can compute (need >= 6 obs)
    full_idx = pd.date_range("2003-01-01", periods=30, freq="MS")
    full_df = pd.DataFrame({"value": np.ones(30) * 5.0}, index=full_idx)
    result = label_regimes(full_df, "UNRATE")
    # Check specifically the 2005 slice
    slice_2005 = result.loc["2005-01-01":"2005-06-01"]
    assert not (slice_2005["regime"] == "recession").any()


def test_regime_labeling_high_volatility() -> None:
    """A stretch with very high rolling std should be labeled 'high_volatility'."""
    rng = np.random.default_rng(0)
    # 60 stable observations followed by 20 very noisy ones
    stable = np.ones(60) * 5.0
    noisy = np.ones(20) * 5.0 + rng.standard_normal(20) * 50.0
    values = np.concatenate([stable, noisy])
    idx = pd.date_range("1995-01-01", periods=80, freq="MS")
    # Keep outside recession dates
    df = pd.DataFrame({"value": values}, index=idx)
    result = label_regimes(df, "CPIAUCSL")
    # At least some of the noisy period should be high_volatility
    high_vol_count = (result.iloc[-10:]["regime"] == "high_volatility").sum()
    assert high_vol_count > 0, "Expected some high_volatility labels in noisy stretch"


# ---------------------------------------------------------------------------
# fetch_series tests
# ---------------------------------------------------------------------------


def test_fetch_series_uses_cache(tmp_path: Path) -> None:
    """fetch_series should return cached data without calling FRED when cache is fresh."""
    # Write a fresh cache file
    sid = "CPIAUCSL"
    idx = pd.date_range("2000-01-01", periods=10, freq="MS")
    df = pd.DataFrame({sid: np.ones(10) * 2.5}, index=idx)
    df.index.name = "date"
    cache_file = tmp_path / f"{sid}.parquet"
    df.to_parquet(cache_file)

    with (
        patch("conformal_econ.data.fred.get_cache_path", return_value=cache_file),
        patch("conformal_econ.data.fred.is_stale", return_value=False),
        patch("conformal_econ.data.fred.read_cache", return_value=df),
        patch("conformal_econ.data.fred._load_from_fred") as mock_fred,
    ):
        result = fetch_series(sid, refresh=False)
        mock_fred.assert_not_called()

    assert len(result) == 10


def test_fetch_series_fallback_no_key(sample_data_dir: Path) -> None:
    """fetch_series should load from sample/ when no API key is set."""
    sid = "UNRATE"

    with (
        patch("conformal_econ.data.fred.load_fred_key", return_value=None),
        patch("conformal_econ.data.fred.is_stale", return_value=True),
        patch("conformal_econ.data.fred.get_cache_path", return_value=Path("/nonexistent/cache.parquet")),
    ):
        result = fetch_series(sid, refresh=True, _sample_dir=sample_data_dir)

    assert len(result) > 0
    assert sid in result.columns
    # UNRATE should have no negatives in sample data
    assert (result[sid] >= 0).all()
