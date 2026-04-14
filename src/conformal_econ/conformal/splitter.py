"""Rolling calibration splits for time-series conformal prediction.

The key invariant throughout: calibration always follows training in time.
No iid random splits. No future information leaking into the calibration set.

Spec: PRD Section 5.3
"""

from __future__ import annotations

import numpy as np


def rolling_calibration_split(
    n: int,
    min_train_frac: float = 0.60,
    cal_frac: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a series of length `n` into temporally ordered train/cal/test sets.

    The minimum training size is max(100, floor(n * min_train_frac)) to ensure
    enough data for model fitting. Calibration immediately follows training.
    Test is everything that remains.

    Args:
        n: Total number of observations.
        min_train_frac: Fraction of series to use as minimum training size.
            Actual train size is max(100, floor(n * min_train_frac)).
        cal_frac: Fraction of series to use as calibration set.

    Returns:
        Tuple of (train_idx, cal_idx, test_idx) as integer index arrays.
        All three are non-overlapping and cover [0, n) in order.

    Raises:
        ValueError: If the series is too short for a meaningful split.
    """
    min_train = max(100, int(n * min_train_frac))
    n_cal = int(n * cal_frac)

    if min_train + n_cal >= n:
        raise ValueError(
            f"Series of length {n} is too short for split: "
            f"train={min_train}, cal={n_cal} leaves no test observations. "
            f"Need at least {min_train + n_cal + 1} observations."
        )

    train_idx = np.arange(min_train)
    cal_idx = np.arange(min_train, min_train + n_cal)
    test_idx = np.arange(min_train + n_cal, n)
    return train_idx, cal_idx, test_idx


class RollingCalibrationSplitter:
    """Generates successive (train_end, cal_end) pairs for rolling re-calibration.

    After the initial split, the training+calibration window expands forward
    by `recal_every` steps at a time. This is how ConformalWrapper.rolling_evaluate()
    re-calibrates as it advances through the test period.

    The initial split uses rolling_calibration_split() with the same fractions.
    Each subsequent split extends both train and cal by recal_every observations.

    Args:
        min_train_frac: Fraction of total series for minimum training size.
        cal_frac: Fraction of total series for calibration window.
        recal_every: Number of test steps between re-calibrations.
    """

    def __init__(
        self,
        min_train_frac: float = 0.60,
        cal_frac: float = 0.20,
        recal_every: int = 12,
    ) -> None:
        self.min_train_frac = min_train_frac
        self.cal_frac = cal_frac
        self.recal_every = recal_every

    def splits(self, n: int) -> list[tuple[int, int, int]]:
        """Return a list of (train_end, cal_end, batch_end) index triples.

        Each triple represents one calibration window. train_end and cal_end
        define the calibration set; batch_end is where the next split starts.
        All indices are exclusive upper bounds (Python slice convention).

        Args:
            n: Total number of observations in the series.

        Returns:
            List of (train_end, cal_end, batch_end) tuples, in temporal order.
        """
        train_idx, cal_idx, test_idx = rolling_calibration_split(
            n, self.min_train_frac, self.cal_frac
        )

        train_end = int(train_idx[-1]) + 1
        cal_end = int(cal_idx[-1]) + 1
        test_start = int(test_idx[0])
        test_end = n

        result = []
        cursor = test_start
        while cursor < test_end:
            batch_end = min(cursor + self.recal_every, test_end)
            result.append((train_end, cal_end, batch_end))
            # Expand the window forward by recal_every
            train_end += self.recal_every
            cal_end += self.recal_every
            cursor = batch_end

        return result
