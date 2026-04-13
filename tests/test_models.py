"""Tests for the model foundation: base class conformance, fit/predict shapes,
Gaussian interval validity, and expected NotImplementedError raises.

All tests use the synthetic AR(1) fixtures from conftest.py — no FRED calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal_econ.models.base import ForecastModel
from conformal_econ.models.statistical import ARIMAModel, ETSModel
from conformal_econ.models.tree import RandomForestModel, XGBoostModel

_HORIZON = 6
_N_TRAIN = 100


@pytest.fixture(scope="module")
def y_train() -> np.ndarray:
    """Short AR(1) training series — 100 obs is enough for model tests."""
    rng = np.random.default_rng(42)
    y = np.zeros(_N_TRAIN)
    for t in range(1, _N_TRAIN):
        y[t] = 0.7 * y[t - 1] + rng.standard_normal()
    # Shift to positive values for ETS multiplicative config
    return y + 10.0


# ---------------------------------------------------------------------------
# Base class conformance
# ---------------------------------------------------------------------------


def test_all_models_implement_base() -> None:
    """Every concrete model must be an instance of ForecastModel."""
    models = [ARIMAModel(), ETSModel(), RandomForestModel(), XGBoostModel()]
    for m in models:
        assert isinstance(m, ForecastModel), f"{m} does not extend ForecastModel"


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------


def test_arima_fit_predict(y_train: np.ndarray) -> None:
    """ARIMA should return a point forecast of the correct shape."""
    m = ARIMAModel()
    m.fit(y_train)
    preds = m.predict(_HORIZON)
    assert preds.shape == (_HORIZON,), f"Expected ({_HORIZON},), got {preds.shape}"
    assert not np.any(np.isnan(preds)), "ARIMA point forecast contains NaN"


def test_arima_gaussian_intervals(y_train: np.ndarray) -> None:
    """ARIMA Gaussian intervals should have correct shape and lower < upper."""
    m = ARIMAModel()
    m.fit(y_train)
    lower, upper = m.predict_gaussian(_HORIZON, alpha=0.10)
    assert lower.shape == (_HORIZON,)
    assert upper.shape == (_HORIZON,)
    assert (upper > lower).all(), "Expected upper > lower for all steps"


def test_arima_name() -> None:
    assert ARIMAModel().name == "ARIMA"


# ---------------------------------------------------------------------------
# ETS
# ---------------------------------------------------------------------------


def test_ets_fit_predict(y_train: np.ndarray) -> None:
    """ETS should return a point forecast of the correct shape."""
    m = ETSModel()
    m.fit(y_train)
    preds = m.predict(_HORIZON)
    assert preds.shape == (_HORIZON,), f"Expected ({_HORIZON},), got {preds.shape}"
    assert not np.any(np.isnan(preds)), "ETS point forecast contains NaN"


def test_ets_gaussian_intervals(y_train: np.ndarray) -> None:
    """ETS simulated intervals should have correct shape and lower < upper."""
    m = ETSModel()
    m.fit(y_train)
    lower, upper = m.predict_gaussian(_HORIZON, alpha=0.10)
    assert lower.shape == (_HORIZON,)
    assert upper.shape == (_HORIZON,)
    assert (upper > lower).all(), "Expected upper > lower for all steps"


def test_ets_name() -> None:
    assert ETSModel().name == "ETS"


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------


def test_rf_fit_predict(y_train: np.ndarray) -> None:
    """RandomForest should return a point forecast of the correct shape."""
    m = RandomForestModel()
    m.fit(y_train)
    preds = m.predict(_HORIZON)
    assert preds.shape == (_HORIZON,), f"Expected ({_HORIZON},), got {preds.shape}"
    assert not np.any(np.isnan(preds)), "RF point forecast contains NaN"


def test_rf_gaussian_raises(y_train: np.ndarray) -> None:
    """RandomForest.predict_gaussian must raise NotImplementedError."""
    m = RandomForestModel()
    m.fit(y_train)
    with pytest.raises(NotImplementedError):
        m.predict_gaussian(_HORIZON, alpha=0.10)


def test_rf_name() -> None:
    assert RandomForestModel().name == "RandomForest"


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


def test_xgb_fit_predict(y_train: np.ndarray) -> None:
    """XGBoost should return a point forecast of the correct shape."""
    m = XGBoostModel()
    m.fit(y_train)
    preds = m.predict(_HORIZON)
    assert preds.shape == (_HORIZON,), f"Expected ({_HORIZON},), got {preds.shape}"
    assert not np.any(np.isnan(preds)), "XGBoost point forecast contains NaN"


def test_xgb_gaussian_raises(y_train: np.ndarray) -> None:
    """XGBoostModel.predict_gaussian must raise NotImplementedError."""
    m = XGBoostModel()
    m.fit(y_train)
    with pytest.raises(NotImplementedError):
        m.predict_gaussian(_HORIZON, alpha=0.10)


def test_xgb_name() -> None:
    assert XGBoostModel().name == "XGBoost"
