"""Gaussian vs conformal interval comparison — the core benchmark result.

For each model × series combination, this module:
1. Splits the series into train / calibration / test using rolling_calibration_split().
2. Fits the model on the training set and generates rolling predictions over the
   calibration set (using the train-fitted model, no per-step refitting) to
   compute calibration residuals → conformal quantile q_hat.
3. Refits the model on train+cal, then generates test-period predictions.
4. Builds conformal intervals (point ± q_hat) and Gaussian intervals (from
   the model's parametric predict_gaussian(), where supported).
5. Slices the test period by economic regime and computes all six evaluation
   metrics for every (method × regime) combination.

The train→cal multi-step forecast used for calibration is a deliberate speed
trade-off: it avoids O(n_cal) refits while still capturing the residual
distribution across the calibration window (including any recession fat-tails).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from conformal_econ.conformal.evaluation import evaluate_all
from conformal_econ.conformal.splitter import rolling_calibration_split
from conformal_econ.models.base import ForecastModel

_DEFAULT_ALPHA = 0.10
_DEFAULT_MIN_TRAIN_FRAC = 0.60
_DEFAULT_CAL_FRAC = 0.20


def _rolling_1step_arima(
    result: object,
    y_full: np.ndarray,
    start: int,
    end: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """1-step-ahead rolling predictions using statsmodels apply().

    Applies the fitted ARIMA coefficients (without re-estimation) to y_full,
    then extracts per-step predictions and Gaussian CIs for positions
    [start, end) using actual observations as conditioning data.

    Args:
        result: Fitted SARIMAXResults object.
        y_full: Full time series (train+cal+test).
        start: First index of the evaluation window (inclusive).
        end: Last index + 1 (exclusive).
        alpha: Significance level for Gaussian CI.

    Returns:
        (point, lower_gauss, upper_gauss) arrays of length (end - start),
        or (point, None, None) if apply() is unavailable.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = result.apply(y_full, refit=False)  # type: ignore[union-attr]
            pred = applied.get_prediction(start=start, end=end - 1, dynamic=False)
            points = np.asarray(pred.predicted_mean)
            ci = np.asarray(pred.conf_int(alpha=alpha))
            return points, ci[:, 0], ci[:, 1]
    except Exception:
        # Fallback: multi-step forecast from start of evaluation window
        multi = result.get_forecast(steps=end - start)  # type: ignore[union-attr]
        points = np.asarray(multi.predicted_mean)
        ci = np.asarray(multi.conf_int(alpha=alpha))
        return points, ci[:, 0], ci[:, 1]


def _rolling_1step_ets(
    result: object,
    y_full: np.ndarray,
    start: int,
    end: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """1-step-ahead rolling predictions using ETS apply() if available.

    Returns (point, lower_gauss, upper_gauss). lower/upper are None when
    ETS simulation fails (non-positive series or convergence issue).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = result.apply(y_full, refit=False)  # type: ignore[union-attr]
            points = np.asarray(applied.fittedvalues)[start:end]
            # Gaussian-style intervals via simulation
            n_sims = 300
            sims = applied.simulate(  # type: ignore[union-attr]
                nsimulations=end - start,
                repetitions=n_sims,
                error="add",
                initial_state=applied.states.smoothed.iloc[start],  # type: ignore[union-attr]
            )
            lower = np.quantile(sims, alpha / 2, axis=1)
            upper = np.quantile(sims, 1 - alpha / 2, axis=1)
            return points, lower, upper
    except Exception:
        # Fallback: forecast + residual std
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            horizon = end - start
            points = np.asarray(result.forecast(steps=horizon))  # type: ignore[union-attr]
            resid_std = float(np.std(result.resid))  # type: ignore[union-attr]
            z = float(np.abs(np.percentile(np.random.standard_normal(5_000), alpha * 50)))
            lower = points - z * resid_std
            upper = points + z * resid_std
            return points, lower, upper


def _tree_1step_preds(
    model: ForecastModel,
    y_full: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    """1-step-ahead predictions for tree models using lag features from actuals.

    Builds the feature vector for each position from the immediately preceding
    actual observations (not recursive predictions). Requires N_LAGS actual
    values to be available before `start`.
    """
    from conformal_econ.models.tree import _N_LAGS, _build_single_row

    points: list[float] = []
    for t in range(start, end):
        recent = y_full[max(0, t - _N_LAGS) : t]
        if len(recent) < _N_LAGS:
            # Pad with the earliest available value
            pad = np.full(_N_LAGS - len(recent), recent[0] if len(recent) else 0.0)
            recent = np.concatenate([pad, recent])
        x = _build_single_row(recent, t)
        # mypy: model has _model attribute for tree models
        pred = float(model._model.predict(x.reshape(1, -1))[0])  # type: ignore[attr-defined]
        points.append(pred)
    return np.array(points)


def _compute_q_hat(
    model: ForecastModel,
    y_full: np.ndarray,
    train_end: int,
    cal_end: int,
    alpha: float,
) -> float:
    """Compute the conformal quantile q_hat from calibration residuals.

    Uses the train-fitted model to generate multi-step predictions across
    the calibration window, then computes residuals against actuals.

    For tree models, uses lag features from actuals for each cal step.
    For ARIMA/ETS, uses the multi-step forecast from end of training.

    Args:
        model: Fitted on y_full[:train_end].
        y_full: Full series.
        train_end: End index of training set (exclusive).
        cal_end: End index of calibration set (exclusive).
        alpha: Significance level.

    Returns:
        Conformal quantile q_hat (non-negative float).
    """
    n_cal = cal_end - train_end
    y_cal = y_full[train_end:cal_end]

    # Tree models: build lag features from actual history
    if hasattr(model, "_model") and hasattr(model._model, "predict"):
        cal_preds = _tree_1step_preds(model, y_full, train_end, cal_end)
    else:
        # ARIMA / ETS: multi-step forecast from end of training
        # predict(n_cal) returns [ŷ_{t+1}, ..., ŷ_{t+n_cal}]
        try:
            raw_result = getattr(model, "_result", None)
            if raw_result is not None:
                pts, _, _ = _rolling_1step_arima(
                    raw_result, y_full[:cal_end], train_end, cal_end, alpha
                )
                cal_preds = pts
            else:
                cal_preds = model.predict(n_cal)
        except Exception:
            cal_preds = model.predict(n_cal)

    cal_residuals = np.abs(y_cal - cal_preds[: len(y_cal)])
    n = len(cal_residuals)
    level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))
    return float(np.quantile(cal_residuals, level))


def compare_model_on_series(
    model: ForecastModel,
    series_id: str,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    regimes: np.ndarray,
    alpha: float = _DEFAULT_ALPHA,
    min_train_frac: float = _DEFAULT_MIN_TRAIN_FRAC,
    cal_frac: float = _DEFAULT_CAL_FRAC,
) -> list[dict]:
    """Evaluate conformal and Gaussian intervals for one model × series.

    Returns a list of metric dicts, one per (method × regime) combination.
    Each dict contains the six evaluation metrics plus metadata columns.

    Args:
        model: Any ForecastModel. predict_gaussian() is called if supported.
        series_id: FRED series identifier (for labeling output rows).
        y: Full time series as a 1-D float array.
        dates: DatetimeIndex aligned with y.
        regimes: String array of regime labels aligned with y.
        alpha: Significance level (0.10 → 90% coverage target).
        min_train_frac: Passed to rolling_calibration_split().
        cal_frac: Passed to rolling_calibration_split().

    Returns:
        List of dicts with keys: series_id, model, regime, method, n_obs,
        coverage, width, winkler, picp, mpiw, coverage_deviation.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    train_idx, cal_idx, test_idx = rolling_calibration_split(n, min_train_frac, cal_frac)
    train_end = int(train_idx[-1]) + 1
    cal_end = int(cal_idx[-1]) + 1

    y_train = y[:train_end]
    y_test = y[cal_end:]
    regimes_test = np.asarray(regimes[cal_end:])

    y_std = float(np.std(y_train)) if np.std(y_train) > 0 else 1.0

    # --- Step 1: calibration → q_hat ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(y_train)
    q_hat = _compute_q_hat(model, y, train_end, cal_end, alpha)

    # --- Step 2: refit on train+cal, get test predictions ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(y[:cal_end])

    n_test = len(y_test)

    # Determine prediction strategy per model type
    raw_result = getattr(model, "_result", None)
    is_arima = raw_result is not None and hasattr(raw_result, "get_forecast")
    is_ets = raw_result is not None and hasattr(raw_result, "forecast") and not is_arima

    if is_arima:
        test_points, lower_gauss, upper_gauss = _rolling_1step_arima(
            raw_result, y, cal_end, n, alpha
        )
        has_gaussian = lower_gauss is not None
    elif is_ets:
        test_points, lower_gauss, upper_gauss = _rolling_1step_ets(
            raw_result, y, cal_end, n, alpha
        )
        has_gaussian = lower_gauss is not None
    elif hasattr(model, "_model") and hasattr(model._model, "predict"):
        # Tree models: lag features from actuals
        test_points = _tree_1step_preds(model, y, cal_end, n)
        lower_gauss = upper_gauss = None
        has_gaussian = False
    else:
        # LSTM or other: use model's predict()
        test_points = model.predict(n_test)
        lower_gauss = upper_gauss = None
        has_gaussian = False
        # Try predict_gaussian as a fallback
        try:
            lower_gauss, upper_gauss = model.predict_gaussian(n_test, alpha)
            has_gaussian = True
        except (NotImplementedError, Exception):
            pass

    test_points = np.asarray(test_points)[:n_test]
    lower_conf = test_points - q_hat
    upper_conf = test_points + q_hat

    if has_gaussian and lower_gauss is not None:
        lower_gauss = np.asarray(lower_gauss)[:n_test]
        upper_gauss = np.asarray(upper_gauss)[:n_test]

    # --- Step 3: per-regime metrics ---
    unique_regimes = list(np.unique(regimes_test))
    all_regime_labels = ["overall"] + unique_regimes

    records: list[dict] = []
    for regime_label in all_regime_labels:
        mask = np.ones(n_test, dtype=bool) if regime_label == "overall" else (regimes_test == regime_label)
        if mask.sum() < 3:
            continue

        y_reg = y_test[mask]
        lo_c, hi_c = lower_conf[mask], upper_conf[mask]
        conf_metrics = evaluate_all(y_reg, lo_c, hi_c, alpha, y_std)

        records.append(
            {
                "series_id": series_id,
                "model": model.name,
                "regime": regime_label,
                "method": "conformal",
                "alpha": alpha,
                "n_obs": int(mask.sum()),
                **conf_metrics,
            }
        )

        if has_gaussian and lower_gauss is not None and upper_gauss is not None:
            lo_g, hi_g = lower_gauss[mask], upper_gauss[mask]
            gauss_metrics = evaluate_all(y_reg, lo_g, hi_g, alpha, y_std)
            records.append(
                {
                    "series_id": series_id,
                    "model": model.name,
                    "regime": regime_label,
                    "method": "gaussian",
                    "alpha": alpha,
                    "n_obs": int(mask.sum()),
                    **gauss_metrics,
                }
            )

    return records
