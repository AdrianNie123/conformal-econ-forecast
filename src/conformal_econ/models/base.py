"""Abstract base class for all forecasting models in this project.

Every model — ARIMA, ETS, Random Forest, XGBoost, LSTM, Chronos-2 — implements
this interface. The conformal wrappers (Week 2) depend only on this interface,
not on any specific model class.

Spec: PRD Section 5.1
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ForecastModel(ABC):
    """Common interface for all point and interval forecasting models.

    Subclasses must implement fit, predict, predict_gaussian, and the name
    property. predict_gaussian should raise NotImplementedError for models
    that have no parametric interval (tree models, LSTM) — conformal wrappers
    handle those through a different code path.
    """

    @abstractmethod
    def fit(self, y_train: np.ndarray) -> None:
        """Fit the model to a univariate training series.

        Args:
            y_train: 1-D array of observations, ordered oldest to newest.
        """

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Generate point forecasts for the next `horizon` steps.

        Args:
            horizon: Number of steps ahead to forecast. Must be >= 1.

        Returns:
            1-D array of length `horizon`.
        """

    @abstractmethod
    def predict_gaussian(
        self, horizon: int, alpha: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate parametric (Gaussian) prediction intervals.

        For models without a parametric form (Random Forest, XGBoost, LSTM),
        raise NotImplementedError — conformal wrappers handle those.

        Args:
            horizon: Number of steps ahead to forecast.
            alpha: Significance level. 0.10 gives a 90% interval.

        Returns:
            Tuple of (lower, upper), each a 1-D array of length `horizon`.

        Raises:
            NotImplementedError: If the model has no parametric interval.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name used in benchmark results and plots."""
