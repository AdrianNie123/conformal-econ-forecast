"""Shared pytest fixtures for the conformal-econ test suite.

All fixtures use synthetic data — no FRED API calls in tests.
Sample parquet files for the FRED fallback tests are created here
as a session-scoped fixture so they exist before test_data.py runs.
"""

from __future__ import annotations

import os
from pathlib import Path

# XGBoost's OpenMP thread pool causes segfaults on macOS when fit() is called
# in a tight loop (e.g., ConformalWrapper calibration). Single-threading fixes it.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pytest


def _ar1_series(n: int, phi: float = 0.8, seed: int = 42) -> np.ndarray:
    """Generate a stationary AR(1) process: y_t = phi * y_{t-1} + eps_t."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.standard_normal()
    return y


@pytest.fixture(scope="session")
def synthetic_monthly_series() -> pd.DataFrame:
    """200-point monthly AR(1) series starting 2000-01."""
    values = _ar1_series(200)
    idx = pd.date_range("2000-01-01", periods=200, freq="MS")
    return pd.DataFrame({"value": values}, index=idx)


@pytest.fixture(scope="session")
def synthetic_quarterly_series() -> pd.DataFrame:
    """80-point quarterly AR(1) series starting 1990-Q1."""
    values = _ar1_series(80, phi=0.6, seed=7)
    idx = pd.date_range("1990-01-01", periods=80, freq="QS")
    return pd.DataFrame({"value": values}, index=idx)


@pytest.fixture(scope="session")
def sample_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped temp directory pre-populated with sample parquet files.

    Used to test the FRED fallback path without needing an API key.
    Each file mimics the structure fetch_series() expects.
    """
    sample_dir = tmp_path_factory.mktemp("sample_data")
    series_ids = ["CPIAUCSL", "UNRATE", "A191RL1Q225SBEA", "DCOILWTICO"]

    for sid in series_ids:
        n = 80
        values = _ar1_series(n, seed=hash(sid) % 2**31)
        # UNRATE must be non-negative
        if sid == "UNRATE":
            values = np.abs(values) + 3.0
        idx = pd.date_range("1990-01-01", periods=n, freq="MS")
        df = pd.DataFrame({sid: values}, index=idx)
        df.index.name = "date"
        df.to_parquet(sample_dir / f"{sid}.parquet")

    return sample_dir


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped temp cache directory for cache write/read tests."""
    return tmp_path_factory.mktemp("cache")
