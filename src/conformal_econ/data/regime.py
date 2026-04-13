"""NBER recession-based regime labeling for economic time series.

Regimes matter because prediction interval coverage degrades in non-stationary
macro environments — recessions and volatility spikes are exactly where Gaussian
intervals break down. This module labels each observation so benchmark results
can be sliced by regime.

Recession dates come directly from NBER's official business cycle chronology.
No API call — hardcoded from https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# NBER recession periods: (peak month, trough month) — both inclusive.
# Source: NBER Business Cycle Dating Committee, accessed 2025.
_NBER_RECESSIONS: list[tuple[str, str]] = [
    ("1960-04", "1961-02"),  # Kennedy recession
    ("1969-12", "1970-11"),  # Nixon recession
    ("1973-11", "1975-03"),  # Oil shock / stagflation
    ("1980-01", "1980-07"),  # Volcker shock I
    ("1981-07", "1982-11"),  # Volcker shock II
    ("1990-07", "1991-03"),  # Gulf War recession
    ("2001-03", "2001-11"),  # Dot-com bust
    ("2007-12", "2009-06"),  # Great Financial Crisis
    ("2020-02", "2020-04"),  # COVID shock
]

# Percentile threshold for "high volatility" regime
_VOL_PERCENTILE = 75
_ROLLING_WINDOW = 12  # months (or quarters if series is quarterly)


def _recession_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Return a boolean Series: True if the date falls within any NBER recession.

    Args:
        index: DatetimeIndex of the series.

    Returns:
        Boolean Series aligned with index.
    """
    mask = pd.Series(False, index=index)
    for start_str, end_str in _NBER_RECESSIONS:
        start = pd.Period(start_str, freq="M").to_timestamp(how="start")
        end = pd.Period(end_str, freq="M").to_timestamp(how="end")
        mask |= (index >= start) & (index <= end)
    return mask


def label_regimes(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """Add a 'regime' column to a series DataFrame.

    Priority order:
    1. recession — date falls within an NBER recession window
    2. high_volatility — rolling std exceeds 75th percentile of historical std
    3. expansion — everything else

    Volatility is computed on the raw values so the threshold reflects the
    series' own scale, not some cross-series normalization.

    Args:
        df: DataFrame with DatetimeIndex and a single value column.
        series_id: FRED series identifier (used to select value column).

    Returns:
        Copy of df with an additional 'regime' column.
    """
    out = df.copy()
    values = out.iloc[:, 0]

    # Rolling std — window scales with series frequency
    # Quarterly series use a shorter window to match monthly's ~1yr
    window = _ROLLING_WINDOW if len(df) > 100 else max(4, len(df) // 10)
    rolling_std = values.rolling(window=window, min_periods=window // 2).std()

    vol_threshold = np.nanpercentile(rolling_std.dropna().values, _VOL_PERCENTILE)

    recession = _recession_mask(out.index)
    high_vol = (rolling_std > vol_threshold) & ~recession

    out["regime"] = "expansion"
    out.loc[high_vol, "regime"] = "high_volatility"
    out.loc[recession, "regime"] = "recession"  # recession wins over high_vol

    return out


def get_recession_bands() -> list[dict[str, str]]:
    """Return NBER recession periods as Plotly-ready shading dicts.

    Each dict has 'start' and 'end' keys as ISO date strings.
    Used by viz modules to overlay recession bands on charts.

    Returns:
        List of {'start': str, 'end': str} dicts.
    """
    bands = []
    for start_str, end_str in _NBER_RECESSIONS:
        bands.append(
            {
                "start": pd.Period(start_str, freq="M")
                .to_timestamp(how="start")
                .isoformat(),
                "end": pd.Period(end_str, freq="M")
                .to_timestamp(how="end")
                .isoformat(),
            }
        )
    return bands
