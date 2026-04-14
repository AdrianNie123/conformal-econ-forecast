"""Coverage, width, Winkler score, PICP, MPIW, and coverage deviation metrics.

All functions are pure numpy — no side effects, no fitting. Each takes aligned
arrays y_true, lower, upper of the same length and returns a scalar.

The headline metric for this project is coverage_deviation: how far the empirical
coverage is from the nominal target (1-alpha). Conformal should be within ±3% of
target in all regimes. Gaussian intervals routinely miss by 15-20% in recessions.

Spec: PRD Section 6.1 — Angelopoulos & Bates (2023) for the coverage guarantee.
"""

from __future__ import annotations

import numpy as np


def empirical_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of true values falling inside the prediction interval.

    This is what the conformal guarantee is about: empirical_coverage should be
    >= 1-alpha with high probability over the calibration randomness.

    Args:
        y_true: Observed values, shape (n,).
        lower: Interval lower bounds, shape (n,).
        upper: Interval upper bounds, shape (n,).

    Returns:
        Fraction in [0, 1] of observations inside [lower, upper].
    """
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean width of prediction intervals: mean(upper - lower).

    Narrower = more informative, all else equal. The trade-off between coverage
    and width is what makes the Winkler score useful.

    Args:
        lower: Interval lower bounds, shape (n,).
        upper: Interval upper bounds, shape (n,).

    Returns:
        Mean interval width. Always >= 0.
    """
    return float(np.mean(upper - lower))


def winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """Winkler score: interval width penalized for misses.

    For each observation t:
        score_t = (upper_t - lower_t)
                + (2/alpha) * max(0, lower_t - y_t)   # penalize undershoot
                + (2/alpha) * max(0, y_t - upper_t)   # penalize overshoot

    Lower is better. A perfect oracle (zero-width, 100% coverage) scores 0.
    Intervals that are wide but cover everything score their mean width.
    Intervals that miss pay the penalty on top of their width.

    Args:
        y_true: Observed values, shape (n,).
        lower: Interval lower bounds, shape (n,).
        upper: Interval upper bounds, shape (n,).
        alpha: Significance level (e.g. 0.10 for 90% intervals).

    Returns:
        Mean Winkler score over all observations.
    """
    width = upper - lower
    penalty_below = np.maximum(0.0, lower - y_true)
    penalty_above = np.maximum(0.0, y_true - upper)
    scores = width + (2.0 / alpha) * (penalty_below + penalty_above)
    return float(np.mean(scores))


def picp(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Prediction Interval Coverage Probability.

    Same computation as empirical_coverage — both measure the fraction of
    observations inside the interval. PICP is the standard name in the
    interval forecasting literature (Khosravi et al., 2011).

    Args:
        y_true: Observed values, shape (n,).
        lower: Interval lower bounds, shape (n,).
        upper: Interval upper bounds, shape (n,).

    Returns:
        PICP in [0, 1].
    """
    return empirical_coverage(y_true, lower, upper)


def mpiw(
    lower: np.ndarray,
    upper: np.ndarray,
    y_std: float,
) -> float:
    """Mean Prediction Interval Width, normalized by series standard deviation.

    MPIW = mean(upper - lower) / std(y_train)

    Normalizing by series std makes MPIW comparable across series with
    different scales. A value of 1.0 means the interval is one standard
    deviation wide on average.

    Args:
        lower: Interval lower bounds, shape (n,).
        upper: Interval upper bounds, shape (n,).
        y_std: Standard deviation of the training series (not the test series,
            to avoid lookahead). Must be > 0.

    Returns:
        Dimensionless MPIW.

    Raises:
        ValueError: If y_std is not positive.
    """
    if y_std <= 0:
        raise ValueError(f"y_std must be positive, got {y_std}")
    return mean_interval_width(lower, upper) / y_std


def coverage_deviation(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """Absolute deviation of empirical coverage from nominal target (1-alpha).

    The project's headline metric. Conformal prediction should be within ±3%
    of target in all regimes. Gaussian intervals in recessions often miss by
    15-20%.

    Args:
        y_true: Observed values, shape (n,).
        lower: Interval lower bounds, shape (n,).
        upper: Interval upper bounds, shape (n,).
        alpha: Significance level. Target coverage is (1 - alpha).

    Returns:
        |empirical_coverage - (1 - alpha)|, in [0, 1].
    """
    target = 1.0 - alpha
    return abs(empirical_coverage(y_true, lower, upper) - target)


def evaluate_all(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
    y_std: float | None = None,
) -> dict[str, float]:
    """Run all six evaluation metrics at once.

    Used by the benchmark runner (Week 3) to collect per-model, per-series,
    per-regime results in one call.

    Args:
        y_true: Observed values, shape (n,).
        lower: Interval lower bounds, shape (n,).
        upper: Interval upper bounds, shape (n,).
        alpha: Significance level (e.g. 0.10 for 90% intervals).
        y_std: Training series std for MPIW normalization. If None, MPIW is
            computed using std(y_true) as an approximation.

    Returns:
        Dict with keys: coverage, width, winkler, picp, mpiw, coverage_deviation.
    """
    std = y_std if y_std is not None else float(np.std(y_true))
    if std == 0.0:
        std = 1.0  # Degenerate series: avoid division by zero, mpiw = width

    return {
        "coverage": empirical_coverage(y_true, lower, upper),
        "width": mean_interval_width(lower, upper),
        "winkler": winkler_score(y_true, lower, upper, alpha),
        "picp": picp(y_true, lower, upper),
        "mpiw": mpiw(lower, upper, std),
        "coverage_deviation": coverage_deviation(y_true, lower, upper, alpha),
    }
