"""Conformal prediction wrappers for all forecasting models.

ConformalWrapper is model-agnostic: it works with any ForecastModel by using
only fit() and predict(). The conformal calibration protocol is manual split
conformal — mathematically equivalent to MAPIE's "base" method but compatible
with non-sklearn models (ARIMA, ETS, LSTM).

The rolling calibration protocol (re-calibrate every 12 steps) is what makes
this honest: the model never sees the test observations during calibration,
and the calibration window moves forward in time alongside the test window.

Spec: PRD Section 5.3
Reference: Angelopoulos & Bates (2023), A Gentle Introduction to Conformal Prediction
"""

from __future__ import annotations

import numpy as np

from conformal_econ.conformal.splitter import (
    RollingCalibrationSplitter,
    rolling_calibration_split,
)
from conformal_econ.models.base import ForecastModel

# Default calibration protocol from PRD Section 5.3
_DEFAULT_ALPHA = 0.10
_DEFAULT_MIN_TRAIN_FRAC = 0.60
_DEFAULT_CAL_FRAC = 0.20
_DEFAULT_RECAL_EVERY = 12


class ConformalWrapper:
    """Wraps any ForecastModel with split conformal prediction intervals.

    The calibration protocol:
    1. Split series into train / cal / test using rolling_calibration_split()
    2. For each t in cal: refit on y[:t], predict 1-step, compute |y[t] - ŷ|
    3. Compute q_hat = quantile((1-alpha)(1 + 1/n_cal)) of calibration scores
    4. Refit on train+cal, predict horizon, return point ± q_hat

    Using 1-step calibration residuals for all forecast horizons is standard
    split conformal practice. It is conservative (intervals are wider than
    necessary for short horizons) but guarantees marginal coverage.

    Args:
        model: Any ForecastModel instance (ARIMA, ETS, RF, XGBoost, LSTM).
        alpha: Significance level. 0.10 targets 90% coverage.
        min_train_frac: Minimum training fraction for rolling_calibration_split.
        cal_frac: Calibration fraction for rolling_calibration_split.
        recal_every: Steps between re-calibrations in rolling_evaluate().
    """

    def __init__(
        self,
        model: ForecastModel,
        alpha: float = _DEFAULT_ALPHA,
        min_train_frac: float = _DEFAULT_MIN_TRAIN_FRAC,
        cal_frac: float = _DEFAULT_CAL_FRAC,
        recal_every: int = _DEFAULT_RECAL_EVERY,
    ) -> None:
        self.model = model
        self.alpha = alpha
        self.min_train_frac = min_train_frac
        self.cal_frac = cal_frac
        self.recal_every = recal_every
        self._q_hat: float | None = None

    def calibrate(self, y_full: np.ndarray) -> None:
        """Compute the conformal quantile from calibration residuals.

        Fits the model on the training portion, computes 1-step-ahead residuals
        on the calibration set (expanding window), then stores q_hat. Finally,
        refits the model on train+cal so it is ready for predict_interval().

        After calibrate(), predict_interval() can be called immediately.

        Args:
            y_full: Full time series, ordered oldest to newest.
        """
        train_idx, cal_idx, _ = rolling_calibration_split(
            len(y_full), self.min_train_frac, self.cal_frac
        )

        # Compute 1-step-ahead calibration scores on expanding window.
        # Each step: fit on y[:t], predict next value, compare to y[t].
        cal_scores: list[float] = []
        for t in cal_idx:
            self.model.fit(y_full[:t])
            pred_1 = float(self.model.predict(1)[0])
            cal_scores.append(abs(float(y_full[t]) - pred_1))

        # Adjusted quantile for finite-sample coverage guarantee.
        # Angelopoulos & Bates (2023), Proposition 1.
        n_cal = len(cal_scores)
        level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / n_cal))
        self._q_hat = float(np.quantile(cal_scores, level))

        # Refit on all data up to end of calibration so predict_interval()
        # uses the most recent observations.
        cal_end = int(cal_idx[-1]) + 1
        self.model.fit(y_full[:cal_end])

    def predict_interval(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """Symmetric conformal intervals: point_forecast ± q_hat.

        Requires calibrate() to have been called first.

        Args:
            horizon: Number of steps ahead to forecast.

        Returns:
            Tuple of (lower, upper), each shape (horizon,).

        Raises:
            RuntimeError: If calibrate() has not been called.
        """
        if self._q_hat is None:
            raise RuntimeError("Call calibrate() before predict_interval().")
        point = self.model.predict(horizon)
        return point - self._q_hat, point + self._q_hat

    def rolling_evaluate(
        self,
        y_full: np.ndarray,
        horizon: int = 1,
    ) -> dict[str, np.ndarray]:
        """Rolling evaluation over the test period with periodic re-calibration.

        Walks through the test period in batches of `recal_every` steps. At the
        start of each batch, re-calibrates on the expanded train+cal window, then
        generates one-step-ahead intervals for each observation in the batch.

        This mimics real deployment: as time passes, new observations are folded
        into the calibration set and the conformal quantile is updated.

        Args:
            y_full: Full time series, ordered oldest to newest.
            horizon: Steps ahead for each prediction. Defaults to 1 (one-step
                rolling evaluation). Use horizon > 1 for multi-step batches.

        Returns:
            Dict with keys 'point', 'lower', 'upper', 'y_true', each a 1-D array
            aligned over the test period observations.
        """
        splitter = RollingCalibrationSplitter(
            min_train_frac=self.min_train_frac,
            cal_frac=self.cal_frac,
            recal_every=self.recal_every,
        )
        split_list = splitter.splits(len(y_full))

        all_point: list[float] = []
        all_lower: list[float] = []
        all_upper: list[float] = []
        all_y: list[float] = []

        for train_end, cal_end, batch_end in split_list:
            # Re-calibrate on the current window.
            # Compute calibration scores on y_full[train_end:cal_end].
            cal_scores: list[float] = []
            for t in range(train_end, cal_end):
                if t >= len(y_full):
                    break
                self.model.fit(y_full[:t])
                pred_1 = float(self.model.predict(1)[0])
                cal_scores.append(abs(float(y_full[t]) - pred_1))

            if cal_scores:
                n_cal = len(cal_scores)
                level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / n_cal))
                q_hat = float(np.quantile(cal_scores, level))
            else:
                # Fallback: use last known q_hat if cal window is empty.
                q_hat = self._q_hat if self._q_hat is not None else 0.0

            # Generate one-step-ahead predictions for each test observation.
            for t in range(cal_end, batch_end):
                if t >= len(y_full):
                    break
                self.model.fit(y_full[:t])
                point = float(self.model.predict(1)[0])
                all_point.append(point)
                all_lower.append(point - q_hat)
                all_upper.append(point + q_hat)
                all_y.append(float(y_full[t]))

        return {
            "point": np.array(all_point),
            "lower": np.array(all_lower),
            "upper": np.array(all_upper),
            "y_true": np.array(all_y),
        }
