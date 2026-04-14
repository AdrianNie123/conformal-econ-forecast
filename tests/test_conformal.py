"""Tests for the conformal prediction core: splitter, wrappers, evaluation, and LSTM.

All tests use synthetic AR(1) data — no FRED API calls. LSTM tests use max_epochs=3
to keep the suite fast without a separate test config.

Week 2 implementation. Spec: PRD Section 5.3 and 6.1.
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal_econ.conformal.evaluation import (
    coverage_deviation,
    empirical_coverage,
    evaluate_all,
    mean_interval_width,
    mpiw,
    picp,
    winkler_score,
)
from conformal_econ.conformal.splitter import (
    RollingCalibrationSplitter,
    rolling_calibration_split,
)
from conformal_econ.conformal.wrappers import ConformalWrapper
from conformal_econ.models.base import ForecastModel
from conformal_econ.models.neural import LSTMModel
from conformal_econ.models.statistical import ARIMAModel, ETSModel
from conformal_econ.models.tree import RandomForestModel, XGBoostModel

_ALPHA = 0.10
_HORIZON = 6


def _ar1(n: int, phi: float = 0.8, seed: int = 0) -> np.ndarray:
    """Stationary AR(1) series for tests."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.standard_normal()
    return y + 10.0  # Shift to positive values for ETS multiplicative config


@pytest.fixture(scope="module")
def y_series() -> np.ndarray:
    """150-point AR(1) series — long enough for rolling_calibration_split."""
    return _ar1(150)


@pytest.fixture(scope="module")
def y_short() -> np.ndarray:
    """80-point AR(1) series — used for LSTM tests where min_train=48 fits."""
    return _ar1(80, seed=7)


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------


def test_rolling_split_temporal_order(y_series: np.ndarray) -> None:
    """train < cal < test — no gaps, no overlaps, strict temporal order."""
    train, cal, test = rolling_calibration_split(len(y_series))
    assert train[-1] < cal[0], "Calibration must start after training ends"
    assert cal[-1] < test[0], "Test must start after calibration ends"


def test_rolling_split_sizes_sum_to_n(y_series: np.ndarray) -> None:
    """train + cal + test == n."""
    n = len(y_series)
    train, cal, test = rolling_calibration_split(n)
    assert len(train) + len(cal) + len(test) == n


def test_rolling_split_no_overlap(y_series: np.ndarray) -> None:
    """Index sets must be disjoint."""
    train, cal, test = rolling_calibration_split(len(y_series))
    assert len(set(train) & set(cal)) == 0, "train ∩ cal must be empty"
    assert len(set(cal) & set(test)) == 0, "cal ∩ test must be empty"


def test_rolling_split_min_train_floor() -> None:
    """Min train is max(100, floor(n * frac)), never below 100 for small series."""
    # n=160, min_train_frac=0.60 → 96 obs, but floor kicks in to 100
    train, _, _ = rolling_calibration_split(160, min_train_frac=0.60)
    assert len(train) == 100


def test_rolling_split_too_short_raises() -> None:
    """Series with only train+cal obs and nothing left for test should raise."""
    with pytest.raises(ValueError, match="too short"):
        rolling_calibration_split(110)  # min_train=100, cal=22 → 122 > 110


def test_rolling_calibration_splitter_temporal_order() -> None:
    """RollingCalibrationSplitter: each split's train_end < cal_end."""
    splitter = RollingCalibrationSplitter()
    splits = splitter.splits(200)
    assert len(splits) > 0
    for train_end, cal_end, batch_end in splits:
        assert train_end < cal_end < batch_end


def test_rolling_calibration_splitter_batch_end_advances() -> None:
    """Each successive batch_end must be strictly greater (time moves forward)."""
    splitter = RollingCalibrationSplitter(recal_every=12)
    splits = splitter.splits(200)
    batch_ends = [s[2] for s in splits]
    assert batch_ends == sorted(batch_ends)
    assert len(set(batch_ends)) == len(batch_ends)  # No duplicates


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def test_empirical_coverage_perfect() -> None:
    """All y inside intervals → coverage = 1.0."""
    y = np.array([1.0, 2.0, 3.0])
    lower = np.array([0.0, 1.0, 2.0])
    upper = np.array([2.0, 3.0, 4.0])
    assert empirical_coverage(y, lower, upper) == 1.0


def test_empirical_coverage_zero() -> None:
    """All y outside intervals → coverage = 0.0."""
    y = np.array([10.0, 20.0, 30.0])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 1.0])
    assert empirical_coverage(y, lower, upper) == 0.0


def test_empirical_coverage_partial() -> None:
    """Half inside → coverage = 0.5."""
    y = np.array([0.5, 5.0])  # 0.5 inside [0,1], 5.0 outside [0,1]
    lower = np.zeros(2)
    upper = np.ones(2)
    assert empirical_coverage(y, lower, upper) == 0.5


def test_mean_interval_width() -> None:
    """Width = mean(upper - lower)."""
    lower = np.array([0.0, 0.0])
    upper = np.array([2.0, 4.0])
    assert mean_interval_width(lower, upper) == 3.0


def test_winkler_score_no_misses() -> None:
    """When all y are covered, Winkler score equals mean interval width."""
    y = np.array([1.0, 1.0])
    lower = np.array([0.0, 0.0])
    upper = np.array([2.0, 2.0])
    expected = mean_interval_width(lower, upper)
    assert winkler_score(y, lower, upper, alpha=0.10) == pytest.approx(expected)


def test_winkler_score_miss_penalty() -> None:
    """A miss by delta adds (2/alpha)*delta to the score."""
    # y=3.0 above upper=2.0, miss = 1.0, alpha=0.10 → penalty = 20.0
    y = np.array([3.0])
    lower = np.array([0.0])
    upper = np.array([2.0])
    width = 2.0
    expected = width + (2.0 / 0.10) * 1.0
    assert winkler_score(y, lower, upper, alpha=0.10) == pytest.approx(expected)


def test_picp_matches_empirical_coverage() -> None:
    """picp() is an alias for empirical_coverage()."""
    y = np.array([0.5, 5.0])
    lower = np.zeros(2)
    upper = np.ones(2)
    assert picp(y, lower, upper) == empirical_coverage(y, lower, upper)


def test_mpiw_normalized() -> None:
    """MPIW = mean_width / y_std."""
    lower = np.array([0.0, 0.0])
    upper = np.array([2.0, 4.0])
    y_std = 2.0
    expected = mean_interval_width(lower, upper) / y_std
    assert mpiw(lower, upper, y_std) == pytest.approx(expected)


def test_mpiw_invalid_std_raises() -> None:
    """mpiw() must raise for non-positive y_std."""
    with pytest.raises(ValueError):
        mpiw(np.zeros(3), np.ones(3), y_std=0.0)


def test_coverage_deviation_zero() -> None:
    """Perfect nominal coverage → deviation = 0."""
    n = 10
    y = np.arange(n, dtype=float)
    # All covered → empirical = 1.0, alpha=0.0 → target = 1.0 → deviation = 0
    lower = y - 1.0
    upper = y + 1.0
    # With alpha=0.0 target is 1.0; empirical_coverage is 1.0
    assert coverage_deviation(y, lower, upper, alpha=0.0) == pytest.approx(0.0)


def test_coverage_deviation_nonzero() -> None:
    """Zero coverage with target 90% → deviation = 0.90."""
    y = np.array([10.0, 10.0])
    lower = np.array([0.0, 0.0])
    upper = np.array([1.0, 1.0])
    assert coverage_deviation(y, lower, upper, alpha=0.10) == pytest.approx(0.90)


def test_evaluate_all_returns_all_keys() -> None:
    """evaluate_all() must return all six metric keys."""
    expected_keys = {"coverage", "width", "winkler", "picp", "mpiw", "coverage_deviation"}
    y = np.array([0.5, 1.5, 2.5])
    lower = np.zeros(3)
    upper = np.full(3, 3.0)
    result = evaluate_all(y, lower, upper, alpha=0.10)
    assert set(result.keys()) == expected_keys


def test_evaluate_all_consistent() -> None:
    """evaluate_all() values must match individual metric functions."""
    y = np.array([1.0, 2.0, 3.0])
    lower = np.array([0.5, 1.5, 2.5])
    upper = np.array([1.5, 2.5, 3.5])
    alpha = 0.10
    result = evaluate_all(y, lower, upper, alpha=alpha)
    assert result["coverage"] == pytest.approx(empirical_coverage(y, lower, upper))
    assert result["width"] == pytest.approx(mean_interval_width(lower, upper))
    assert result["winkler"] == pytest.approx(winkler_score(y, lower, upper, alpha))


# ---------------------------------------------------------------------------
# ConformalWrapper
# ---------------------------------------------------------------------------


def test_conformal_wrapper_calibrate_sets_q_hat(y_series: np.ndarray) -> None:
    """After calibrate(), q_hat must be set and positive."""
    wrapper = ConformalWrapper(ARIMAModel(), alpha=_ALPHA)
    wrapper.calibrate(y_series)
    assert wrapper._q_hat is not None
    assert wrapper._q_hat > 0.0, "Conformal quantile must be positive"


def test_conformal_wrapper_interval_shape(y_series: np.ndarray) -> None:
    """predict_interval(h) returns two arrays of shape (h,)."""
    wrapper = ConformalWrapper(ARIMAModel(), alpha=_ALPHA)
    wrapper.calibrate(y_series)
    lower, upper = wrapper.predict_interval(_HORIZON)
    assert lower.shape == (_HORIZON,), f"Expected ({_HORIZON},), got {lower.shape}"
    assert upper.shape == (_HORIZON,), f"Expected ({_HORIZON},), got {upper.shape}"


def test_conformal_wrapper_lower_lt_upper(y_series: np.ndarray) -> None:
    """lower < upper for all forecast steps."""
    wrapper = ConformalWrapper(ARIMAModel(), alpha=_ALPHA)
    wrapper.calibrate(y_series)
    lower, upper = wrapper.predict_interval(_HORIZON)
    assert (upper > lower).all(), "upper must be strictly greater than lower"


def test_conformal_wrapper_no_nan(y_series: np.ndarray) -> None:
    """predict_interval output must not contain NaN."""
    wrapper = ConformalWrapper(ETSModel(), alpha=_ALPHA)
    wrapper.calibrate(y_series)
    lower, upper = wrapper.predict_interval(_HORIZON)
    assert not np.any(np.isnan(lower)), "lower contains NaN"
    assert not np.any(np.isnan(upper)), "upper contains NaN"


def test_conformal_wrapper_requires_calibrate(y_series: np.ndarray) -> None:
    """predict_interval before calibrate raises RuntimeError."""
    wrapper = ConformalWrapper(ARIMAModel())
    with pytest.raises(RuntimeError, match="calibrate"):
        wrapper.predict_interval(_HORIZON)


@pytest.mark.parametrize(
    "model_cls",
    [ARIMAModel, ETSModel, RandomForestModel, XGBoostModel],
)
def test_conformal_wrapper_all_models(
    model_cls: type[ForecastModel], y_series: np.ndarray
) -> None:
    """ConformalWrapper works end-to-end with each model."""
    wrapper = ConformalWrapper(model_cls(), alpha=_ALPHA)
    wrapper.calibrate(y_series)
    lower, upper = wrapper.predict_interval(_HORIZON)
    assert lower.shape == (_HORIZON,)
    assert (upper > lower).all()


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------


def test_lstm_implements_base() -> None:
    """LSTMModel must be an instance of ForecastModel."""
    assert isinstance(LSTMModel(), ForecastModel)


def test_lstm_name() -> None:
    assert LSTMModel().name == "LSTM"


def test_lstm_fit_predict_shape(y_short: np.ndarray) -> None:
    """LSTM fit + predict should return shape (horizon,) with no NaN."""
    m = LSTMModel(lookback=12, max_epochs=3, patience=2)
    m.fit(y_short)
    preds = m.predict(_HORIZON)
    assert preds.shape == (_HORIZON,), f"Expected ({_HORIZON},), got {preds.shape}"
    assert not np.any(np.isnan(preds)), "LSTM prediction contains NaN"


def test_lstm_predict_without_fit_raises() -> None:
    """predict() before fit() raises RuntimeError."""
    m = LSTMModel()
    with pytest.raises(RuntimeError, match="fit"):
        m.predict(_HORIZON)


def test_lstm_gaussian_raises(y_short: np.ndarray) -> None:
    """predict_gaussian must raise NotImplementedError."""
    m = LSTMModel(lookback=12, max_epochs=3, patience=2)
    m.fit(y_short)
    with pytest.raises(NotImplementedError):
        m.predict_gaussian(_HORIZON, alpha=0.10)


def test_lstm_with_conformal_wrapper(y_series: np.ndarray) -> None:
    """LSTM + ConformalWrapper end-to-end: intervals have correct shape.

    Uses y_series (150 obs) rather than y_short because ConformalWrapper
    requires at least 100 training observations (the hard floor in splitter).
    """
    lstm = LSTMModel(lookback=12, max_epochs=3, patience=2)
    wrapper = ConformalWrapper(lstm, alpha=_ALPHA)
    wrapper.calibrate(y_series)
    lower, upper = wrapper.predict_interval(_HORIZON)
    assert lower.shape == (_HORIZON,)
    assert (upper > lower).all()


def test_lstm_too_short_raises() -> None:
    """fit() on a series shorter than lookback+2 raises ValueError."""
    m = LSTMModel(lookback=24)
    with pytest.raises(ValueError, match="too short"):
        m.fit(np.ones(10))
