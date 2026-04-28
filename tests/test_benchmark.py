"""Tests for the benchmark compare and runner modules.

All tests use tiny synthetic series to keep runtime under 30 seconds.
The actual benchmark results (results/benchmark_results.parquet) are
validated separately via the runner's CLI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_econ.benchmark.compare import compare_model_on_series
from conformal_econ.models.statistical import ARIMAModel, ETSModel
from conformal_econ.models.tree import RandomForestModel, XGBoostModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def short_series() -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """200-point monthly AR(1) series with three regime labels."""
    rng = np.random.default_rng(0)
    n = 200
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.8 * y[t - 1] + rng.standard_normal() * 0.5
    y += 10.0  # shift to positive (required by ETS multiplicative)

    dates = pd.date_range("2000-01-01", periods=n, freq="MS")

    regimes = np.full(n, "expansion")
    # Inject a recession stretch (30 obs) and high-vol stretch (20 obs)
    regimes[80:110] = "recession"
    regimes[140:160] = "high_volatility"

    return y, dates, regimes


# ---------------------------------------------------------------------------
# compare_model_on_series tests
# ---------------------------------------------------------------------------

class TestCompareModelOnSeries:
    """Tests for the per-model comparison function."""

    def test_arima_returns_records(self, short_series):
        y, dates, regimes = short_series
        records = compare_model_on_series(
            ARIMAModel(), "TEST", y, dates, regimes, alpha=0.10
        )
        assert len(records) > 0
        # ARIMA has Gaussian intervals, so we expect both methods
        methods = {r["method"] for r in records}
        assert "conformal" in methods
        assert "gaussian" in methods

    def test_ets_returns_records(self, short_series):
        y, dates, regimes = short_series
        records = compare_model_on_series(
            ETSModel(), "TEST", y, dates, regimes, alpha=0.10
        )
        assert len(records) > 0
        methods = {r["method"] for r in records}
        assert "conformal" in methods

    def test_random_forest_conformal_only(self, short_series):
        y, dates, regimes = short_series
        records = compare_model_on_series(
            RandomForestModel(), "TEST", y, dates, regimes, alpha=0.10
        )
        methods = {r["method"] for r in records}
        assert "conformal" in methods
        assert "gaussian" not in methods

    def test_xgboost_conformal_only(self, short_series):
        y, dates, regimes = short_series
        records = compare_model_on_series(
            XGBoostModel(), "TEST", y, dates, regimes, alpha=0.10
        )
        methods = {r["method"] for r in records}
        assert "conformal" in methods
        assert "gaussian" not in methods

    def test_regime_rows_present(self, short_series):
        y, dates, regimes = short_series
        records = compare_model_on_series(
            ARIMAModel(), "TEST", y, dates, regimes, alpha=0.10
        )
        regime_labels = {r["regime"] for r in records}
        assert "overall" in regime_labels

    def test_metrics_in_valid_range(self, short_series):
        y, dates, regimes = short_series
        records = compare_model_on_series(
            ARIMAModel(), "TEST", y, dates, regimes, alpha=0.10
        )
        for rec in records:
            assert 0.0 <= rec["coverage"] <= 1.0, f"coverage out of range: {rec['coverage']}"
            assert rec["width"] >= 0.0, f"width negative: {rec['width']}"
            assert rec["winkler"] >= 0.0, f"winkler negative: {rec['winkler']}"
            assert 0.0 <= rec["picp"] <= 1.0, f"picp out of range: {rec['picp']}"
            assert rec["mpiw"] >= 0.0, f"mpiw negative: {rec['mpiw']}"
            assert rec["coverage_deviation"] >= 0.0, f"deviation negative: {rec['coverage_deviation']}"

    def test_n_obs_positive(self, short_series):
        y, dates, regimes = short_series
        records = compare_model_on_series(
            ETSModel(), "TEST", y, dates, regimes, alpha=0.10
        )
        for rec in records:
            assert rec["n_obs"] >= 3, f"n_obs too small: {rec['n_obs']}"

    def test_conformal_coverage_roughly_nominal(self, short_series):
        """Conformal coverage should be in [0.70, 1.0] at alpha=0.10."""
        y, dates, regimes = short_series
        records = compare_model_on_series(
            ARIMAModel(), "TEST", y, dates, regimes, alpha=0.10
        )
        overall_conf = next(
            (r for r in records if r["regime"] == "overall" and r["method"] == "conformal"),
            None,
        )
        assert overall_conf is not None
        # Conformal guarantee: coverage >= 1-alpha with high probability.
        # With 200 points and 20% test set (40 obs), allow some slack.
        assert overall_conf["coverage"] >= 0.70, (
            f"Conformal coverage too low: {overall_conf['coverage']:.3f}"
        )

    def test_alpha_stored_in_records(self, short_series):
        y, dates, regimes = short_series
        records = compare_model_on_series(
            ETSModel(), "TEST", y, dates, regimes, alpha=0.05
        )
        for rec in records:
            assert rec["alpha"] == 0.05

    def test_series_id_stored(self, short_series):
        y, dates, regimes = short_series
        records = compare_model_on_series(
            ARIMAModel(), "MY_SERIES", y, dates, regimes, alpha=0.10
        )
        for rec in records:
            assert rec["series_id"] == "MY_SERIES"

    def test_model_name_stored(self, short_series):
        y, dates, regimes = short_series
        records = compare_model_on_series(
            RandomForestModel(), "TEST", y, dates, regimes, alpha=0.10
        )
        for rec in records:
            assert rec["model"] == "RandomForest"

    def test_short_series_raises_on_too_small(self):
        """Series too short for train/cal/test split should raise ValueError."""
        y = np.arange(50, dtype=float)
        dates = pd.date_range("2000-01-01", periods=50, freq="MS")
        regimes = np.full(50, "expansion")
        with pytest.raises(ValueError, match="too short"):
            compare_model_on_series(ARIMAModel(), "SHORT", y, dates, regimes)


# ---------------------------------------------------------------------------
# runner smoke test
# ---------------------------------------------------------------------------

class TestRunnerSmoke:
    """Smoke tests for the full benchmark runner."""

    def test_run_benchmark_arima_one_series(self, tmp_path, short_series):
        """Runner should produce a valid parquet with at least one result row."""
        from unittest.mock import patch

        import numpy as np
        import pandas as pd

        from conformal_econ.benchmark.runner import run_benchmark
        from conformal_econ.data.regime import label_regimes

        y, dates, regimes_arr = short_series
        df = pd.DataFrame({"CPIAUCSL": y}, index=dates)
        df.index.name = "date"
        df_with_regimes = label_regimes(df, "CPIAUCSL")

        def fake_fetch(sid, refresh=False):
            return df

        output_path = tmp_path / "test_results.parquet"

        with patch("conformal_econ.benchmark.runner.fetch_series", side_effect=fake_fetch):
            results = run_benchmark(
                series_ids=["CPIAUCSL"],
                model_names=["ARIMA"],
                alpha=0.10,
                output=output_path,
                verbose=False,
            )

        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert output_path.exists()

        required_cols = {
            "series_id", "model", "regime", "method", "n_obs",
            "coverage", "width", "winkler", "picp", "mpiw", "coverage_deviation",
        }
        assert required_cols.issubset(set(results.columns))

    def test_run_benchmark_missing_series_skipped(self, tmp_path):
        """Runner should skip series that fail to load rather than crashing."""
        from conformal_econ.benchmark.runner import run_benchmark

        output_path = tmp_path / "results.parquet"
        # CPIAUCSL will fail if sample data doesn't exist and no API key
        # Use a minimal run that forces an error
        import unittest.mock as mock

        def always_fail(sid, refresh=False):
            raise FileNotFoundError("no data")

        with mock.patch("conformal_econ.benchmark.runner.fetch_series", side_effect=always_fail):
            with pytest.raises(RuntimeError, match="no results"):
                run_benchmark(
                    series_ids=["CPIAUCSL"],
                    model_names=["ARIMA"],
                    output=output_path,
                    verbose=False,
                )
