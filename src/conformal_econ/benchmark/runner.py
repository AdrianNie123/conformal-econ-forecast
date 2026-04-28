"""Full benchmark pipeline — conformal vs Gaussian on all four FRED series.

Runs ARIMA, ETS, Random Forest, and XGBoost on CPIAUCSL, UNRATE,
A191RL1Q225SBEA, and DCOILWTICO. Produces a parquet file with per-model ×
per-series × per-regime metrics for both conformal and Gaussian intervals.

The benchmark uses sample data from data/sample/ if no FRED_API_KEY is set,
so it runs offline in CI and demo environments without any API dependency.

Usage:
    python -m conformal_econ.benchmark.runner
    python -m conformal_econ.benchmark.runner --series CPIAUCSL UNRATE
    python -m conformal_econ.benchmark.runner --models ARIMA ETS --alpha 0.05
    python -m conformal_econ.benchmark.runner --output results/bench.parquet
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import pandas as pd

from conformal_econ.benchmark.compare import compare_model_on_series
from conformal_econ.data.fred import SERIES, fetch_series
from conformal_econ.data.regime import label_regimes
from conformal_econ.models.statistical import ARIMAModel, ETSModel
from conformal_econ.models.tree import RandomForestModel, XGBoostModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_OUTPUT = _PROJECT_ROOT / "results" / "benchmark_results.parquet"

# Models available for benchmarking. LSTM excluded by default (too slow for demo runs).
_ALL_MODELS = ["ARIMA", "ETS", "RandomForest", "XGBoost"]

_MODEL_FACTORIES = {
    "ARIMA": ARIMAModel,
    "ETS": ETSModel,
    "RandomForest": RandomForestModel,
    "XGBoost": XGBoostModel,
}

# Series human-readable names for the results table
_SERIES_NAMES = {sid: meta["name"] for sid, meta in SERIES.items()}


def run_benchmark(
    series_ids: list[str] | None = None,
    model_names: list[str] | None = None,
    alpha: float = 0.10,
    output: Path | None = None,
    refresh: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full conformal vs Gaussian benchmark.

    Loads each series (from cache, FRED API, or sample/ fallback), applies
    regime labels, then runs compare_model_on_series() for every model × series
    pair. Results are aggregated into a single DataFrame and optionally saved
    to a parquet file.

    Args:
        series_ids: FRED series IDs to include. Defaults to all four.
        model_names: Model names to run. Defaults to ARIMA, ETS, RF, XGBoost.
        alpha: Significance level (0.10 = 90% coverage target).
        output: Path to write results parquet. Defaults to results/benchmark_results.parquet.
        refresh: Force re-fetch from FRED (bypasses cache).
        verbose: Print progress to stdout.

    Returns:
        DataFrame with columns: series_id, series_name, model, regime, method,
        alpha, n_obs, coverage, width, winkler, picp, mpiw, coverage_deviation.
    """
    if series_ids is None:
        series_ids = list(SERIES.keys())
    if model_names is None:
        model_names = _ALL_MODELS
    if output is None:
        output = _DEFAULT_OUTPUT

    all_records: list[dict] = []
    t0_total = time.time()

    for sid in series_ids:
        if verbose:
            print(f"\n[{sid}] {_SERIES_NAMES.get(sid, sid)}")

        # Load series
        try:
            df = fetch_series(sid, refresh=refresh)
        except Exception as exc:
            if verbose:
                print(f"  ERROR loading {sid}: {exc}")
            continue

        # Apply regime labels and drop NaNs
        df_regimes = label_regimes(df, sid)
        df_regimes = df_regimes.dropna(subset=[sid])
        y = df_regimes[sid].values.astype(float)
        dates = df_regimes.index
        regimes = df_regimes["regime"].values

        if verbose:
            print(f"  {len(y)} observations, regimes: {dict(pd.Series(regimes).value_counts())}")

        for model_name in model_names:
            t0 = time.time()
            if verbose:
                print(f"  [{model_name}] ...", end="", flush=True)

            factory = _MODEL_FACTORIES.get(model_name)
            if factory is None:
                if verbose:
                    print(f" SKIP (unknown model)")
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = factory()
                    records = compare_model_on_series(
                        model=model,
                        series_id=sid,
                        y=y,
                        dates=dates,
                        regimes=regimes,
                        alpha=alpha,
                    )

                # Attach series name
                for rec in records:
                    rec["series_name"] = _SERIES_NAMES.get(sid, sid)

                all_records.extend(records)
                elapsed = time.time() - t0
                if verbose:
                    n_rows = len(records)
                    print(f" done ({n_rows} rows, {elapsed:.1f}s)")

            except Exception as exc:
                elapsed = time.time() - t0
                if verbose:
                    print(f" ERROR after {elapsed:.1f}s: {exc}")
                continue

    if not all_records:
        raise RuntimeError("Benchmark produced no results. Check series and model configuration.")

    results = pd.DataFrame(all_records)

    # Reorder columns for readability
    front_cols = ["series_id", "series_name", "model", "regime", "method", "alpha", "n_obs"]
    metric_cols = ["coverage", "width", "winkler", "picp", "mpiw", "coverage_deviation"]
    results = results[front_cols + metric_cols]

    # Save to parquet
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output, index=False)

    elapsed_total = time.time() - t0_total
    if verbose:
        print(f"\nBenchmark complete: {len(results)} rows in {elapsed_total:.1f}s → {output}")

    return results


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run the conformal vs Gaussian benchmark on FRED economic series."
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=list(SERIES.keys()),
        choices=list(SERIES.keys()),
        help="FRED series IDs to include (default: all four).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=_ALL_MODELS,
        choices=list(_MODEL_FACTORIES.keys()),
        help="Models to run (default: all four).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Significance level (default: 0.10 → 90%% coverage).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output parquet path (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-fetch from FRED, ignoring cache.",
    )
    args = parser.parse_args()

    run_benchmark(
        series_ids=args.series,
        model_names=args.models,
        alpha=args.alpha,
        output=args.output,
        refresh=args.refresh,
        verbose=True,
    )


if __name__ == "__main__":
    _cli()
