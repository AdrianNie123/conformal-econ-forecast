"""ARIMA and ETS models with Gaussian prediction intervals.

These are the baseline parametric models. They get proper Gaussian intervals
from statsmodels — those are what conformal prediction is supposed to outperform
in regimes where the normality assumption breaks down.

ARIMA: grid search over (p, d, q) via AIC to avoid manual order selection.
ETS: tries additive vs multiplicative error/trend/seasonal configurations.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from conformal_econ.models.base import ForecastModel


class ARIMAModel(ForecastModel):
    """ARIMA with automatic order selection via AIC grid search.

    Searches p in [0, 3], d in [0, 2], q in [0, 3]. Picks the combination
    with the lowest AIC. Convergence failures are silently skipped.

    Gaussian intervals come from statsmodels' built-in get_forecast().conf_int(),
    which assumes normally distributed residuals — valid in expansion regimes,
    much less so in recessions.
    """

    def __init__(self) -> None:
        self._result: object | None = None
        self._best_order: tuple[int, int, int] | None = None

    def fit(self, y_train: np.ndarray) -> None:
        """Grid search ARIMA orders and fit the best model.

        Args:
            y_train: 1-D float array, ordered oldest to newest.
        """
        best_aic = float("inf")
        best_result = None
        best_order = (1, 1, 0)

        p_range = range(0, 4)
        d_range = range(0, 3)
        q_range = range(0, 4)

        for p, d, q in itertools.product(p_range, d_range, q_range):
            if p == 0 and q == 0:
                continue  # intercept-only — not useful
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = ARIMA(y_train, order=(p, d, q))
                    result = m.fit()
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_result = result
                    best_order = (p, d, q)
            except Exception:
                continue

        if best_result is None:
            # Fallback to ARIMA(1,1,0) if grid search finds nothing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = ARIMA(y_train, order=(1, 1, 0))
                best_result = m.fit()
            best_order = (1, 1, 0)

        self._result = best_result
        self._best_order = best_order

    def predict(self, horizon: int) -> np.ndarray:
        """Point forecasts from the fitted ARIMA model.

        Args:
            horizon: Steps ahead to forecast.

        Returns:
            1-D array of length `horizon`.
        """
        assert self._result is not None, "Call fit() before predict()"
        fc = self._result.get_forecast(steps=horizon)  # type: ignore[union-attr]
        # predicted_mean is a pandas Series when fit on Series, numpy array otherwise
        return np.asarray(fc.predicted_mean)

    def predict_gaussian(
        self, horizon: int, alpha: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gaussian prediction intervals from ARIMA's built-in forecast.

        statsmodels uses alpha as the significance level (0.10 = 90% CI),
        which matches our convention throughout the project.

        Args:
            horizon: Steps ahead.
            alpha: Significance level (e.g. 0.10 for 90% coverage).

        Returns:
            Tuple of (lower, upper) arrays.
        """
        assert self._result is not None, "Call fit() before predict_gaussian()"
        fc = self._result.get_forecast(steps=horizon)  # type: ignore[union-attr]
        ci = fc.conf_int(alpha=alpha)
        # conf_int returns a DataFrame when fit on Series, ndarray otherwise
        ci_arr = np.asarray(ci)
        return ci_arr[:, 0], ci_arr[:, 1]

    @property
    def name(self) -> str:
        return "ARIMA"


class ETSModel(ForecastModel):
    """Exponential smoothing (Holt-Winters) with Gaussian prediction intervals.

    Tries additive and multiplicative error/trend configurations, picking the
    fit with the lowest AIC. Seasonality is left unspecified — series frequency
    varies and we're not fitting seasonal patterns at this stage.

    Gaussian intervals use statsmodels' simulate() method: generates 500 paths
    from the fitted error distribution and takes quantiles. This is technically
    still parametric (it assumes the fitted error distribution is correct), but
    it's a reasonable approximation and honest about what ETS actually does.
    """

    def __init__(self) -> None:
        self._result: object | None = None

    def fit(self, y_train: np.ndarray) -> None:
        """Fit ETS, selecting the best error/trend configuration by AIC.

        Args:
            y_train: 1-D float array, ordered oldest to newest.
        """
        # Ensure all positive for multiplicative — fall back to additive if not
        all_positive = bool((y_train > 0).all())

        # statsmodels ExponentialSmoothing doesn't expose error type directly;
        # we vary trend and damped_trend to cover the main ETS configurations.
        trend_configs: list[tuple[str | None, bool]] = [
            ("add", False),
            ("add", True),
            (None, False),
        ]
        if all_positive:
            trend_configs += [("mul", False), ("mul", True)]

        best_aic = float("inf")
        best_result = None

        for trend, damped in trend_configs:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = ExponentialSmoothing(
                        y_train,
                        trend=trend,
                        damped_trend=damped,
                        seasonal=None,
                        initialization_method="estimated",
                    )
                    result = m.fit(optimized=True, remove_bias=True)
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_result = result
            except Exception:
                continue

        if best_result is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = ExponentialSmoothing(y_train, trend="add", seasonal=None)
                best_result = m.fit(optimized=True)

        self._result = best_result

    def predict(self, horizon: int) -> np.ndarray:
        """Point forecasts from the fitted ETS model.

        Args:
            horizon: Steps ahead to forecast.

        Returns:
            1-D array of length `horizon`.
        """
        assert self._result is not None, "Call fit() before predict()"
        return np.asarray(self._result.forecast(steps=horizon))  # type: ignore[union-attr]

    def predict_gaussian(
        self, horizon: int, alpha: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate prediction intervals from the fitted ETS error distribution.

        Runs 500 simulation paths forward and takes alpha/2 and 1-alpha/2
        quantiles. Not closed-form, but it's what statsmodels' ETS actually
        supports and it's honest about the uncertainty structure.

        Args:
            horizon: Steps ahead.
            alpha: Significance level (e.g. 0.10 for 90% coverage).

        Returns:
            Tuple of (lower, upper) arrays of length `horizon`.
        """
        assert self._result is not None, "Call fit() before predict_gaussian()"

        n_sims = 500
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # simulate() returns shape (nsimulations, repetitions)
                sims = self._result.simulate(  # type: ignore[union-attr]
                    nsimulations=horizon,
                    repetitions=n_sims,
                    error="add",
                )
            lower = np.quantile(sims, alpha / 2, axis=1)
            upper = np.quantile(sims, 1 - alpha / 2, axis=1)
        except Exception:
            # Fallback: use point forecast ± 1.96 * in-sample residual std
            point = self.predict(horizon)
            resid_std = float(
                np.std(self._result.resid)  # type: ignore[union-attr]
            )
            z = float(np.abs(np.percentile(np.random.standard_normal(10_000), alpha * 50)))
            lower = point - z * resid_std
            upper = point + z * resid_std

        return lower, upper

    @property
    def name(self) -> str:
        return "ETS"
