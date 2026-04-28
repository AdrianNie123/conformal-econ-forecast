"""Generate synthetic sample data for all four FRED series.

Produces realistic synthetic economic time series that match the statistical
characteristics of the real FRED data (scale, autocorrelation, regime
heteroscedasticity) without requiring a live API key.

Regime-specific volatility is critical: during NBER recessions the synthetic
shocks are larger, so Gaussian intervals fitted on calm training data will
undercover recession observations. That's the benchmark's central demonstration.

Output: data/sample/{SERIES_ID}.parquet for each of the four series.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# NBER recession dates (same as regime.py)
# ---------------------------------------------------------------------------

_NBER_RECESSIONS: list[tuple[str, str]] = [
    ("1960-04", "1961-02"),
    ("1969-12", "1970-11"),
    ("1973-11", "1975-03"),
    ("1980-01", "1980-07"),
    ("1981-07", "1982-11"),
    ("1990-07", "1991-03"),
    ("2001-03", "2001-11"),
    ("2007-12", "2009-06"),
    ("2020-02", "2020-04"),
]


def _recession_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    mask = np.zeros(len(dates), dtype=bool)
    for start_str, end_str in _NBER_RECESSIONS:
        start = pd.Period(start_str, freq="M").to_timestamp(how="start")
        end = pd.Period(end_str, freq="M").to_timestamp(how="end")
        mask |= (dates >= start) & (dates <= end)
    return mask


# ---------------------------------------------------------------------------
# Series generators
# ---------------------------------------------------------------------------

def _gen_cpiaucsl(rng: np.random.Generator) -> pd.DataFrame:
    """CPI All-Urban Consumers (monthly, 1960-01 to 2024-12).

    Starts ~29, reaches ~316 by 2024 with regime-specific inflation rates
    and heteroscedastic noise.
    """
    dates = pd.date_range("1960-01-01", "2024-12-01", freq="MS")
    n = len(dates)
    rec = _recession_mask(dates)

    # Monthly inflation rate baseline (varies by decade)
    monthly_rate = np.full(n, 0.0025)
    for i, d in enumerate(dates):
        if 1966 <= d.year <= 1982:
            monthly_rate[i] = 0.0065  # stagflation era
        elif 2021 <= d.year <= 2022:
            monthly_rate[i] = 0.0062  # post-COVID inflation

    # Heteroscedastic noise: recessions → supply shocks → higher volatility
    noise_std = np.where(rec, 0.005, 0.002)
    monthly_rate += rng.normal(0, noise_std)
    monthly_rate = np.clip(monthly_rate, -0.015, 0.020)

    level = np.zeros(n)
    level[0] = 29.0
    for i in range(1, n):
        level[i] = level[i - 1] * (1.0 + monthly_rate[i])

    df = pd.DataFrame({"CPIAUCSL": level}, index=dates)
    df.index.name = "date"
    return df


def _gen_unrate(rng: np.random.Generator) -> pd.DataFrame:
    """Unemployment Rate (monthly, 1960-01 to 2024-12).

    Mean-reverting AR(1) with recession spikes and fat-tailed shocks.
    """
    dates = pd.date_range("1960-01-01", "2024-12-01", freq="MS")
    n = len(dates)
    rec = _recession_mask(dates)

    phi = 0.96
    unrate = np.zeros(n)
    unrate[0] = 5.2

    for i in range(1, n):
        mean_target = 8.5 if rec[i] else 5.0
        shock_std = 0.45 if rec[i] else 0.12
        unrate[i] = phi * unrate[i - 1] + (1 - phi) * mean_target + rng.normal(0, shock_std)
        unrate[i] = np.clip(unrate[i], 2.5, 15.0)

    # COVID spike
    covid_idx = np.where((dates.year == 2020) & (dates.month == 4))[0]
    if len(covid_idx):
        unrate[covid_idx[0]] = 14.7

    df = pd.DataFrame({"UNRATE": unrate}, index=dates)
    df.index.name = "date"
    return df


def _gen_gdp(rng: np.random.Generator) -> pd.DataFrame:
    """Real GDP Growth, annualized % (quarterly, 1960-Q1 to 2024-Q3).

    AR(1) with fat-tailed recession shocks to break ARIMA's normality assumption.
    """
    dates = pd.date_range("1960-01-01", "2024-10-01", freq="QS")
    n = len(dates)

    # Quarter-frequency recession mask
    rec = np.zeros(n, dtype=bool)
    for start_str, end_str in _NBER_RECESSIONS:
        start = pd.Period(start_str, freq="M").to_timestamp(how="start")
        end = pd.Period(end_str, freq="M").to_timestamp(how="end")
        rec |= (dates >= start) & (dates <= end)

    phi = 0.45
    gdp = np.zeros(n)
    gdp[0] = 3.5

    for i in range(1, n):
        if rec[i]:
            mean_g = -3.5
            shock_std = 4.5
        else:
            mean_g = 3.0
            shock_std = 1.2
        gdp[i] = phi * gdp[i - 1] + (1 - phi) * mean_g + rng.normal(0, shock_std)
        gdp[i] = np.clip(gdp[i], -35.0, 12.0)

    # COVID shock Q2 2020
    covid_idx = np.where((dates.year == 2020) & (dates.quarter == 2))[0]
    if len(covid_idx):
        gdp[covid_idx[0]] = -29.9

    df = pd.DataFrame({"A191RL1Q225SBEA": gdp}, index=dates)
    df.index.name = "date"
    return df


def _gen_oil(rng: np.random.Generator) -> pd.DataFrame:
    """WTI Crude Oil Price, $/barrel (monthly, 1986-01 to 2024-12).

    Log-random-walk with drift and regime-specific volatility. Oil is the
    most non-Gaussian of the four series — exactly where conformal shines.
    """
    dates = pd.date_range("1986-01-01", "2024-12-01", freq="MS")
    n = len(dates)
    rec = _recession_mask(dates)

    log_price = np.zeros(n)
    log_price[0] = np.log(26.0)  # ~$26 in Jan 1986

    for i in range(1, n):
        drift = 0.003
        vol = 0.13 if rec[i] else 0.045
        log_price[i] = log_price[i - 1] + drift + rng.normal(0, vol)

    price = np.exp(log_price)

    # Landmark events — normalize to approximate real magnitudes
    idx_2008 = np.where((dates.year == 2008) & (dates.month == 7))[0]
    if len(idx_2008):
        scale = 133.0 / price[idx_2008[0]]
        price *= scale

    # COVID crash (Apr 2020)
    idx_covid = np.where((dates.year == 2020) & (dates.month == 4))[0]
    if len(idx_covid):
        price[idx_covid[0]] = max(price[idx_covid[0] - 1] * 0.30, 10.0)

    price = np.clip(price, 5.0, 200.0)

    df = pd.DataFrame({"DCOILWTICO": price}, index=dates)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    sample_dir = project_root / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    generators = {
        "CPIAUCSL": _gen_cpiaucsl,
        "UNRATE": _gen_unrate,
        "A191RL1Q225SBEA": _gen_gdp,
        "DCOILWTICO": _gen_oil,
    }

    for sid, gen_fn in generators.items():
        df = gen_fn(rng)
        out_path = sample_dir / f"{sid}.parquet"
        df.to_parquet(out_path)
        print(f"  {sid}: {len(df)} obs → {out_path}")


if __name__ == "__main__":
    main()
