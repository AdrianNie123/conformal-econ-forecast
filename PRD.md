# PRD: Conformal Forecast — Economic Uncertainty Quantification Toolkit

**Version:** 1.0  
**Author:** Adriannie  
**Status:** Ready for build  
**Target timeline:** 3–5 weeks  
**Deployment:** Streamlit (local + Streamlit Cloud)  
**Publishing:** GitHub (public, showcase-quality)

---

## 1. Project summary

Build a Python toolkit and interactive dashboard that wraps six forecasting methods — ARIMA, ETS, Random Forest, XGBoost, LSTM, and Chronos-2 — with conformal prediction intervals, then benchmarks all methods on curated economic time series across distinct macroeconomic regimes. The core thesis: Gaussian prediction intervals lie. Conformal prediction doesn't. Show this, systematically, on real data.

This is a portfolio showcase. It must be technically rigorous, visually exceptional, and reproducible end-to-end.

---

## 2. Goals

### Primary
- Demonstrate that conformal prediction intervals achieve valid empirical coverage where Gaussian intervals fail
- Benchmark six forecasting methods on four economic series across three regime types
- Ship a clean, installable Python package with a compelling Streamlit dashboard
- Publish a showcase-quality GitHub repo that signals advanced ML × econometrics skill

### Non-goals (explicitly out of scope)
- Real-time data feeds or live inference API
- User authentication or multi-user sessions
- Forecasting beyond 12-step horizons
- Causal inference or treatment effect estimation (separate project)

---

## 3. Technical architecture

```
conformal-econ-forecast/
│
├── src/
│   └── conformal_econ/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── fred.py          # FRED API ingestion
│       │   ├── regime.py        # Regime labeling (NBER dates)
│       │   └── cache.py         # Local parquet caching
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py          # Abstract ForecastModel class
│       │   ├── statistical.py   # ARIMA, ETS (statsmodels)
│       │   ├── tree.py          # Random Forest, XGBoost
│       │   ├── neural.py        # LSTM (PyTorch)
│       │   └── foundation.py    # Chronos-2 (HuggingFace)
│       ├── conformal/
│       │   ├── __init__.py
│       │   ├── wrappers.py      # MAPIE wrappers for all models
│       │   ├── splitter.py      # Rolling/expanding calibration splits
│       │   └── evaluation.py   # Coverage, width, WINKLER score
│       ├── benchmark/
│       │   ├── __init__.py
│       │   ├── runner.py        # Full benchmark pipeline
│       │   └── compare.py      # Gaussian vs conformal comparison
│       └── viz/
│           ├── __init__.py
│           ├── forecast_plot.py
│           ├── coverage_plot.py
│           └── regime_plot.py
│
├── app/
│   ├── main.py                  # Streamlit entry point
│   ├── pages/
│   │   ├── 01_overview.py
│   │   ├── 02_explorer.py       # Interactive series + model selector
│   │   ├── 03_benchmark.py      # Full benchmark results table
│   │   ├── 04_regimes.py        # Regime breakdown analysis
│   │   └── 05_methodology.py    # Math + framework explainer
│   └── components/
│       ├── sidebar.py
│       ├── metrics_card.py
│       └── regime_badge.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_validation.ipynb
│   └── 03_results_analysis.ipynb
│
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_conformal.py
│   └── test_benchmark.py
│
├── data/
│   └── cache/                   # gitignored, local parquet files
│
├── results/
│   └── benchmark_results.parquet  # Committed pre-run results
│
├── .github/
│   └── workflows/
│       ├── ci.yml               # pytest + ruff + mypy on push
│       └── data_refresh.yml     # Weekly FRED data pull
│
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
├── LICENSE                      # MIT
└── README.md                    # Showcase-grade (see Section 8)
```

---

## 4. Data specification

### 4.1 Economic series (FRED API)

| Series | FRED ID | Frequency | Start | Regimes |
|--------|---------|-----------|-------|---------|
| CPI Inflation (YoY) | `CPIAUCSL` | Monthly | 1960 | All |
| Unemployment Rate | `UNRATE` | Monthly | 1960 | All |
| Real GDP Growth | `A191RL1Q225SBEA` | Quarterly | 1960 | All |
| WTI Crude Oil Price | `DCOILWTICO` | Monthly | 1986 | All |

### 4.2 Regime labeling

Use NBER recession dates (hardcoded from official NBER data, no API dependency):

- **Recession**: NBER expansion dummy = 0
- **Expansion**: NBER expansion dummy = 1, volatility below 75th percentile
- **High volatility**: Rolling 12-month std > 75th percentile of historical std

Label each observation with its regime. All benchmark results are reported globally and broken down by regime.

### 4.3 Data pipeline rules
- All raw data cached to `data/cache/` as parquet (gitignored)
- Cache invalidation: 7 days or manual `--refresh` flag
- FRED API key loaded from `.env` via `python-dotenv`, never hardcoded
- If no API key present: load from `data/sample/` (committed small sample for CI)
- All series validated on load: no negative unemployment, no missing > 5% consecutive

---

## 5. Models specification

### 5.1 Abstract base class (all models must implement)

```python
class ForecastModel(ABC):
    def fit(self, y_train: np.ndarray) -> None: ...
    def predict(self, horizon: int) -> np.ndarray: ...
    def predict_gaussian(self, horizon: int, alpha: float) -> tuple[np.ndarray, np.ndarray]: ...
    @property
    def name(self) -> str: ...
```

### 5.2 Model implementations

**ARIMA** (`statsmodels.tsa.arima.model.ARIMA`)
- Auto order selection via AIC minimization over p∈[0,3], d∈[0,2], q∈[0,3]
- Gaussian intervals: model's built-in `get_forecast()` confidence intervals
- Conformal: MAPIE `TimeSeriesSplit` with rolling calibration

**ETS** (`statsmodels.tsa.holtwinters.ExponentialSmoothing`)
- Auto model selection (additive/multiplicative error, trend, seasonality)
- Gaussian intervals: analytical prediction intervals from statsmodels
- Conformal: same as ARIMA

**Random Forest** (`sklearn.ensemble.RandomForestRegressor`)
- Features: lags 1–12, rolling mean (3, 6, 12), rolling std (6), month dummies
- Hyperparameters: n_estimators=500, max_depth=10, min_samples_leaf=5
- No Gaussian intervals (non-parametric baseline)
- Conformal: MAPIE `EnbPI` (Ensemble Batch Prediction Intervals)

**XGBoost** (`xgboost.XGBRegressor`)
- Same feature engineering as Random Forest
- Hyperparameters: n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8
- No Gaussian intervals
- Conformal: MAPIE `EnbPI`

**LSTM** (`torch.nn.LSTM`)
- Architecture: 2 layers, hidden_size=64, dropout=0.2
- Input: 24-step lookback window
- Training: Adam optimizer, lr=1e-3, early stopping patience=10
- Conformal: MAPIE with rolling window calibration
- Device: auto-detect CUDA/MPS/CPU

**Chronos-2** (`autogluon/chronos-t5-small` via HuggingFace)
- Use `chronos-t5-small` (410M params) for speed; configurable to `large`
- Native probabilistic output: extract quantiles directly (no MAPIE wrapping)
- Quantiles at α/2 and 1-α/2 for fair comparison
- Cache model weights locally after first download

### 5.3 Conformal wrapper specification

All conformal intervals use **α = 0.1** (target 90% coverage) as default, user-configurable in the dashboard to 0.05 and 0.20.

Rolling calibration protocol:
- Minimum training size: max(100 observations, 60% of series)
- Calibration set: 20% of training data (rolling window, not random split)
- Test evaluation: remaining observations
- Re-calibration: every 12 steps (rolling forward)

This is the critical methodological contribution — honest rolling calibration that respects the time-series structure, not iid conformal splits.

---

## 6. Benchmark specification

### 6.1 Evaluation metrics (per model, per series, per regime)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Empirical coverage** | % observations inside interval | Should equal 1-α (e.g. 90%) |
| **Interval width** | Mean(upper - lower) | Narrower = more informative |
| **Winkler score** | Width + 2/α × penalty for misses | Lower = better (combines both) |
| **PICP** | Prediction Interval Coverage Probability | Same as empirical coverage |
| **MPIW** | Mean Prediction Interval Width | Normalized by series std |
| **Coverage deviation** | |empirical - target| | Should be near 0 |

### 6.2 The core comparison

For each model × series × regime:
1. Gaussian interval coverage (where applicable)
2. Conformal interval coverage
3. Coverage deviation from nominal (90%)

The headline result: a table/chart showing Gaussian intervals systematically under-cover during recessions and volatility spikes, while conformal intervals maintain valid coverage.

### 6.3 Benchmark runner

```bash
python -m conformal_econ.benchmark.runner \
  --series all \
  --models all \
  --alpha 0.1 \
  --output results/benchmark_results.parquet
```

Results saved to parquet. Dashboard loads pre-computed results — no re-running required for the demo.

---

## 7. Streamlit dashboard specification

### 7.1 Visual design direction

**Aesthetic:** Editorial / data journalism. Think Bloomberg Terminal meets academic paper. Dark background (`#0A0F1E`), off-white type (`#F0EDE8`), electric blue accent (`#00D4FF`), amber for warnings/recessions (`#FFB347`), clean monospaced figures.

**Typography:**
- Display: `IBM Plex Serif` (headlines, hero numbers)
- Body: `IBM Plex Sans` (labels, descriptions)  
- Data: `IBM Plex Mono` (all metrics, tick labels, code)

**Color system:**
```python
COLORS = {
    "bg": "#0A0F1E",
    "surface": "#141929",
    "border": "#1E2A45",
    "text_primary": "#F0EDE8",
    "text_muted": "#8892A4",
    "accent_blue": "#00D4FF",
    "accent_amber": "#FFB347",
    "success": "#00E5A0",
    "danger": "#FF4D6D",
    "recession_band": "rgba(255, 179, 71, 0.08)",
}
```

**Chart library:** Plotly (dark theme, custom template). All charts: transparent background, thin gridlines, no chart borders.

**Layout:** Wide mode, custom CSS injected via `st.markdown`. Sidebar for global controls. Pages for sections.

### 7.2 Page specifications

**Page 1 — Overview (`01_overview.py`)**
- Hero section: "Conformal Forecast" title + one-sentence thesis
- Three metric cards: number of series, models benchmarked, headline coverage result
- Animated "coverage deviation" bar chart: Gaussian vs Conformal across all regimes
- Navigation hint to other pages

**Page 2 — Interactive Explorer (`02_explorer.py`)**
- Sidebar controls: series selector, model selector, alpha slider (0.05/0.10/0.20), regime filter
- Main chart: time series with conformal bands shaded, recession bands behind
- Below chart: rolling empirical coverage line (should hover around 90%)
- Metrics panel: current coverage, mean width, Winkler score — formatted as large numbers
- Toggle: "Show Gaussian intervals" (overlay comparison, red dashed lines)

**Page 3 — Full Benchmark (`03_benchmark.py`)**
- Styled dataframe: all models × all series × global metrics
- Heatmap: coverage deviation by model × regime (green = valid, red = invalid)
- Scatter: width vs coverage colored by model
- Sortable, filterable — Streamlit's `st.dataframe` with column config

**Page 4 — Regime Analysis (`04_regimes.py`)**
- Side-by-side bar charts: coverage by regime for each model
- Key insight callout box: "Gaussian ARIMA covers 73% in recessions vs target 90%"
- Recession timeline: NBER shaded bands on a full series chart
- Pull-quote styled callout: the headline finding stated plainly

**Page 5 — Methodology (`05_methodology.py`)**
- LaTeX-rendered equations (st.latex): conformal prediction score function, coverage guarantee theorem
- Step-by-step protocol: how calibration was done
- Citations block: Angelopoulos & Bates, MAPIE paper, StatsForecast
- Code snippet: 10-line example of how to use the package

### 7.3 Custom CSS (inject via st.markdown)

Minimum required injections:
- Font imports (Google Fonts: IBM Plex family)
- Background color override
- Sidebar background
- Metric card styling (border, padding, accent line)
- Hide Streamlit default footer and hamburger menu
- Custom scrollbar

---

## 8. README specification (showcase-grade)

The README is part of the deliverable. Structure:

```markdown
# Conformal Forecast

> When Gaussian intervals lie, conformal prediction tells the truth.

[badges: Python 3.11+, License MIT, Code style ruff, Tests passing]

[1 screenshot of the dashboard — dark, elegant]

## The problem
[3 sentences: Gaussian intervals assume normality. Economies are not normal.
Recessions, crises, and volatility spikes systematically break coverage.
Conformal prediction is distribution-free and provably valid.]

## Results
[Headline table: method × regime × coverage — the smoking gun]

## Quick start
[5-line install + run block]

## Dashboard
[Animated GIF of dashboard or second screenshot]

## Architecture
[Folder tree, brief description of each module]

## Methods
[Bullet list of 6 models, link to methodology page in app]

## Data
[FRED series used, regime labeling methodology]

## Reproducing results
[Command to run benchmark from scratch]

## Citation & references
[Angelopoulos & Bates 2023, MAPIE, StatsForecast]

## License
MIT
```

---

## 9. Security and best practices

### 9.1 Secrets management
- FRED API key: `.env` file only, loaded with `python-dotenv`
- `.env` listed in `.gitignore` (verified before first commit)
- `.env.example` committed with placeholder: `FRED_API_KEY=your_key_here`
- No API keys in notebooks, no hardcoded credentials anywhere
- CI uses GitHub Secrets for `FRED_API_KEY`

### 9.2 Code quality
- **Formatter:** `ruff format` (replaces black)
- **Linter:** `ruff check` with rules: E, F, I, N, UP, B, SIM
- **Type checking:** `mypy` on `src/` (strict mode for core modules)
- **Pre-commit hooks:** ruff + mypy run on every commit
- Line length: 88 characters
- All public functions: docstrings (Google style)
- No `# type: ignore` without inline explanation

### 9.3 Testing
- Framework: `pytest` with `pytest-cov`
- Coverage target: ≥ 80% on `src/conformal_econ/`
- Test data: small synthetic series (no FRED API call in tests)
- Fixtures: shared in `conftest.py`
- CI: runs on push to `main` and all PRs

### 9.4 Dependency management
```toml
# pyproject.toml — pinned minor versions, not patch
[project]
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26,<2.0",
    "pandas>=2.1,<3.0",
    "statsmodels>=0.14,<1.0",
    "scikit-learn>=1.4,<2.0",
    "xgboost>=2.0,<3.0",
    "torch>=2.2,<3.0",
    "MAPIE>=0.8,<1.0",
    "transformers>=4.40,<5.0",
    "streamlit>=1.32,<2.0",
    "plotly>=5.20,<6.0",
    "fredapi>=0.5,<1.0",
    "python-dotenv>=1.0,<2.0",
    "pyarrow>=15.0,<16.0",
]
```

### 9.5 Git hygiene
- `.gitignore`: `data/cache/`, `.env`, `__pycache__`, `.mypy_cache`, `.ruff_cache`, `*.pyc`, `*.egg-info`, `dist/`, `.venv/`
- Branch strategy: `main` (protected) + feature branches
- Commit messages: conventional commits (`feat:`, `fix:`, `docs:`, `test:`)
- No large binary files committed (model weights, large datasets)

---

## 10. CI/CD (GitHub Actions)

### `.github/workflows/ci.yml`
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: ruff check src/ tests/
      - run: ruff format --check src/ tests/
      - run: mypy src/conformal_econ/
      - run: pytest tests/ --cov=conformal_econ --cov-report=term-missing
    env:
      FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
```

### `.github/workflows/data_refresh.yml`
```yaml
name: Weekly data refresh
on:
  schedule:
    - cron: "0 6 * * 1"   # Every Monday 6am UTC
  workflow_dispatch:
jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e .
      - run: python -m conformal_econ.data.fred --refresh
      - run: python -m conformal_econ.benchmark.runner --output results/benchmark_results.parquet
      - uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "chore: weekly data and benchmark refresh"
    env:
      FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
```

---

## 11. Build sequence for Claude Code

Execute in this exact order. Start each session with a fresh `/clear`.

### Week 1 — Foundation
1. Scaffold full project structure (`pyproject.toml`, all `__init__.py`, `.gitignore`, `.env.example`)
2. Implement `data/fred.py` with FRED API ingestion, caching, validation
3. Implement `data/regime.py` with NBER recession labeling
4. Write `tests/test_data.py` — verify with synthetic data first
5. Implement `models/base.py` abstract class
6. Implement `models/statistical.py` (ARIMA + ETS)
7. Implement `models/tree.py` (RF + XGBoost)
8. Write `tests/test_models.py`

### Week 2 — Conformal core
1. Implement `conformal/wrappers.py` — MAPIE integration for all models
2. Implement `conformal/splitter.py` — rolling calibration protocol
3. Implement `conformal/evaluation.py` — all metrics (coverage, width, Winkler)
4. Write `tests/test_conformal.py`
5. Implement `models/neural.py` (LSTM)
6. Implement `models/foundation.py` (Chronos-2)
7. Run first end-to-end test on one series

### Week 3 — Benchmark + viz
1. Implement `benchmark/runner.py` — full pipeline
2. Implement `benchmark/compare.py` — Gaussian vs conformal
3. Run full benchmark, save `results/benchmark_results.parquet`
4. Implement all `viz/` modules (Plotly, dark theme)
5. Validate all results — check coverage numbers make sense

### Week 4 — Dashboard + polish
1. Build `app/main.py` — Streamlit shell, custom CSS, page routing
2. Build each page (01 → 05), wiring to viz modules
3. Design and implement all metric cards and callout components
4. Implement regime badges and recession band overlays
5. Mobile responsiveness check (wide mode + sidebar collapse)

### Week 5 — Publishing
1. Write showcase README with screenshots
2. Set up GitHub Actions (ci.yml + data_refresh.yml)
3. Add pre-commit config
4. Final pass: docstrings, type hints, inline comments
5. Tag `v1.0.0`, push to GitHub public repo
6. Deploy to Streamlit Community Cloud (connect repo, set secret)

---

## 12. Success criteria

The project is done when:

- [ ] `pip install -e .` works cleanly on Python 3.11
- [ ] `pytest tests/` passes with ≥80% coverage
- [ ] `ruff check` and `mypy` pass with zero errors
- [ ] Full benchmark runs end-to-end in < 30 minutes on CPU
- [ ] Dashboard loads in < 3 seconds from pre-computed results
- [ ] Coverage results: conformal intervals within ±3% of target in all regimes
- [ ] Gaussian intervals visibly under-cover in recession regime (< 80% vs 90% target)
- [ ] GitHub repo is public with showcase README and CI badge showing green
- [ ] No API keys or secrets committed anywhere in git history
