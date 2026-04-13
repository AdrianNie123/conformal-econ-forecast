"""Random Forest and XGBoost models for economic time series.

Tree models don't produce parametric intervals — that's the point. They're
non-parametric learners that need conformal wrapping (Week 2) to get valid
coverage. predict_gaussian raises NotImplementedError here by design.

Feature engineering: lags 1-12, rolling mean (3, 6, 12-period), rolling
std (6-period), and cyclical month encoding. Multi-step forecasting uses
recursive prediction — each step feeds the previous prediction back as a lag.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from conformal_econ.models.base import ForecastModel

_N_LAGS = 12


def _build_lag_features(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 1-D series into a supervised learning dataset.

    Each row is one observation. Features are lags 1-12, rolling means
    (3, 6, 12-step), rolling std (6-step), and cyclical position (i % 12)
    to capture approximate seasonality without needing the actual calendar.

    Args:
        series: 1-D float array, ordered oldest to newest.

    Returns:
        Tuple of (X, y) where X has shape (n - n_lags, n_features)
        and y has shape (n - n_lags,).
    """
    n = len(series)
    x_rows = []
    y_rows = []

    for i in range(_N_LAGS, n):
        row: list[float] = []

        # Lags 1–12
        for lag in range(1, _N_LAGS + 1):
            row.append(float(series[i - lag]))

        # Rolling means
        row.append(float(np.mean(series[i - 3 : i])))
        row.append(float(np.mean(series[i - 6 : i])))
        row.append(float(np.mean(series[i - 12 : i])))

        # Rolling std (6-step)
        row.append(float(np.std(series[i - 6 : i])))

        # Approximate seasonality — position within a 12-step cycle
        row.append(float(i % 12))

        x_rows.append(row)
        y_rows.append(float(series[i]))

    return np.array(x_rows, dtype=np.float64), np.array(y_rows, dtype=np.float64)


def _build_single_row(recent: np.ndarray, step_idx: int) -> np.ndarray:
    """Build one feature row from the most recent `_N_LAGS` values.

    Used during recursive multi-step prediction: `recent` is always
    the last _N_LAGS observations (real or already-predicted).

    Args:
        recent: 1-D array of the last _N_LAGS values.
        step_idx: Position in the forecast horizon (for cyclical feature).

    Returns:
        1-D feature array matching the column layout from _build_lag_features.
    """
    assert len(recent) >= _N_LAGS

    row: list[float] = []

    # Lags 1–12 (most recent first)
    for lag in range(1, _N_LAGS + 1):
        row.append(float(recent[-lag]))

    row.append(float(np.mean(recent[-3:])))
    row.append(float(np.mean(recent[-6:])))
    row.append(float(np.mean(recent[-12:])))
    row.append(float(np.std(recent[-6:])))
    row.append(float(step_idx % 12))

    return np.array(row, dtype=np.float64)


class RandomForestModel(ForecastModel):
    """Random Forest regressor for time series, using recursive multi-step forecasting.

    Hyperparameters from PRD Section 5.2:
    n_estimators=500, max_depth=10, min_samples_leaf=5.

    No Gaussian intervals — use conformal wrappers (MAPIE EnbPI) for coverage.
    """

    def __init__(self) -> None:
        self._model = RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )
        self._y_train: np.ndarray | None = None

    def fit(self, y_train: np.ndarray) -> None:
        """Fit Random Forest on lag features from the training series.

        Args:
            y_train: 1-D float array, ordered oldest to newest.
        """
        self._y_train = y_train.copy()
        x_feat, y_feat = _build_lag_features(y_train)
        self._model.fit(x_feat, y_feat)

    def predict(self, horizon: int) -> np.ndarray:
        """Recursive multi-step forecast.

        Each step uses the previous _N_LAGS values (real + predicted) as input.

        Args:
            horizon: Steps ahead to forecast.

        Returns:
            1-D array of length `horizon`.
        """
        assert self._y_train is not None, "Call fit() before predict()"

        history = list(self._y_train[-_N_LAGS:])
        predictions: list[float] = []

        for step in range(horizon):
            x = _build_single_row(np.array(history), step)
            pred = float(self._model.predict(x.reshape(1, -1))[0])
            predictions.append(pred)
            history.append(pred)
            history.pop(0)

        return np.array(predictions, dtype=np.float64)

    def predict_gaussian(
        self, horizon: int, alpha: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Not implemented — Random Forest has no parametric interval.

        Use MAPIE EnbPI conformal wrapper instead (conformal/wrappers.py).
        """
        raise NotImplementedError(
            "RandomForest has no parametric prediction interval. "
            "Use conformal wrappers from conformal_econ.conformal.wrappers."
        )

    @property
    def name(self) -> str:
        return "RandomForest"


class XGBoostModel(ForecastModel):
    """XGBoost regressor for time series, using recursive multi-step forecasting.

    Hyperparameters from PRD Section 5.2:
    n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8.

    No Gaussian intervals — use conformal wrappers for coverage.
    """

    def __init__(self) -> None:
        self._model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbosity=0,
        )
        self._y_train: np.ndarray | None = None

    def fit(self, y_train: np.ndarray) -> None:
        """Fit XGBoost on lag features from the training series.

        Args:
            y_train: 1-D float array, ordered oldest to newest.
        """
        self._y_train = y_train.copy()
        x_feat, y_feat = _build_lag_features(y_train)
        self._model.fit(x_feat, y_feat)

    def predict(self, horizon: int) -> np.ndarray:
        """Recursive multi-step forecast.

        Args:
            horizon: Steps ahead to forecast.

        Returns:
            1-D array of length `horizon`.
        """
        assert self._y_train is not None, "Call fit() before predict()"

        history = list(self._y_train[-_N_LAGS:])
        predictions: list[float] = []

        for step in range(horizon):
            x = _build_single_row(np.array(history), step)
            pred = float(self._model.predict(x.reshape(1, -1))[0])
            predictions.append(pred)
            history.append(pred)
            history.pop(0)

        return np.array(predictions, dtype=np.float64)

    def predict_gaussian(
        self, horizon: int, alpha: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Not implemented — XGBoost has no parametric interval.

        Use MAPIE EnbPI conformal wrapper instead.
        """
        raise NotImplementedError(
            "XGBoost has no parametric prediction interval. "
            "Use conformal wrappers from conformal_econ.conformal.wrappers."
        )

    @property
    def name(self) -> str:
        return "XGBoost"
