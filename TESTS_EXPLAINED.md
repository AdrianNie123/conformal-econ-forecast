# Tests, Explained

This document walks through every test in this project in plain English. It's
written for someone who understands economics but hasn't spent much time in a
codebase. If you know what a recession is and have a rough sense of what a
statistical model does, you have everything you need to follow along.

---

## What tests are and why we have them

A test is a short piece of code that checks whether the main code does what it's
supposed to do. You write the test once, and then every time you change something,
you run all the tests to make sure you didn't break anything you weren't looking at.

The analogy I keep coming back to is a pre-flight checklist. A pilot who flew
perfectly yesterday still runs through it before takeoff today. Not because they
distrust themselves, but because "it worked last time" is not a safety guarantee.
Code is the same. You add a new feature in one file, and something three files
away quietly breaks. Tests catch that before it matters.

This project has 61 tests across three files. They cover the data pipeline, the
forecasting models, and the conformal prediction machinery. Run them all with
`pytest tests/` — if everything is green, the code is behaving as designed. If
something fails, you know exactly what broke and where.

---

## The models, plain English

Before the tests make sense, the models need to. Here's what each one actually does.

**ARIMA** (Autoregressive Integrated Moving Average) reads the recent history of a
series and extrapolates it forward. It looks for patterns — "this series tends to
rise after two months of decline" — and uses them to build a forecast. Think of it
like projecting next month's rent based on the trajectory of the last 12 months.
It works well when the world is stable. It falls apart during crises, because it
assumes the future looks like the recent past, and a recession looks nothing like
the expansion that preceded it. That failure is exactly what this project measures.

**ETS** (Exponential Smoothing) also works from history, but with a different logic.
It takes a weighted average of all past observations, where recent ones count more
than old ones. The intuition: if you're trying to guess tomorrow's inflation rate,
last month matters more than five years ago. ETS is simpler than ARIMA but
surprisingly competitive on real economic data, especially for series without
strong trend breaks.

**Random Forest** doesn't think in terms of time series structure at all. Instead,
it engineers a set of features from the data — lags, rolling averages, rolling
volatility — and trains 500 independent decision trees on those features. Each
tree votes on a forecast, and the forest averages the votes. It's like polling 500
economists who have all seen the same spreadsheet and asking for the consensus.
No single tree is reliable; collectively, they're hard to beat on many datasets.

**XGBoost** also uses the engineered features, but trains differently. It builds
trees sequentially: the first tree makes a rough forecast, the second tree
specifically tries to fix the first one's mistakes, the third fixes the second's
mistakes, and so on for 300 rounds. Think of it as a manuscript going through
rounds of editorial revision where each pass targets what the previous pass got
wrong. XGBoost tends to win on structured tabular data, which is why it's
dominated applied ML competitions for the last decade.

**LSTM** (Long Short-Term Memory) is a neural network designed to work with
sequences. Unlike the tree models, it doesn't see a flat feature row — it sees
a window of the last 24 observations in order and reads through them the way you
read a sentence. Earlier words in a sentence inform the meaning of later ones.
Earlier observations in an economic series inform what comes next. The LSTM is
trained to learn which parts of that history to remember and which to forget.
It's the most computationally expensive model here and the hardest to explain,
but it's also the one most capable of picking up on nonlinear temporal patterns
that ARIMA would miss entirely.

---

## `tests/test_data.py` — 12 tests

These tests cover the data infrastructure: saving and loading cached data,
validating that incoming data is clean, labeling economic regimes, and making
sure the data fetching logic doesn't call the FRED API when it doesn't need to.

None of these tests use real FRED data. They all generate synthetic series on
the fly so the tests run identically with or without an API key.

### Cache

---

**`test_cache_write_read`**

*What is this test asking?* If I save a DataFrame to disk and read it back, do I
get exactly what I saved?

The test creates a small table of fake monthly data, writes it to a temporary
directory as a Parquet file, reads it back, and checks that the values match.
The one wrinkle worth knowing: Parquet stores timestamps but not pandas frequency
metadata (like "month start"), so the test explicitly skips checking that
attribute and focuses on the values themselves.

*Passing means:* the read-back DataFrame matches the original. Data round-trips
correctly through the cache layer.

*Industry relevance:* Every production ML pipeline caches data. Knowing that your
cache layer is lossless is foundational — silent data corruption in caching is
one of the harder bugs to track down after the fact.

---

**`test_cache_staleness_fresh`**

*What is this test asking?* Does the code correctly recognize that a file written
a moment ago is not stale?

It writes a file, immediately checks whether it's "stale" using a 7-day window,
and expects the answer to be no.

*Passing means:* a freshly written cache file is not flagged for refresh.

*Industry relevance:* Cache invalidation is famously one of the two hard problems
in computer science. Getting the freshness check right is what prevents you from
either re-fetching data you just downloaded or serving stale data you downloaded
two weeks ago.

---

**`test_cache_staleness_old`**

*What is this test asking?* Does the code correctly recognize that a file written
8 days ago is stale?

It creates a file and manually sets its modification timestamp to 8 days in the
past, then checks whether it's stale with a 7-day window. Expects yes.

*Passing means:* old files are correctly flagged so the pipeline knows to refresh
them.

*Industry relevance:* FRED updates its series on a rolling basis. A pipeline that
doesn't detect stale data will silently work on outdated numbers — a real problem
if you're using unemployment data that's been revised since you last pulled it.

---

**`test_cache_miss_returns_none`**

*What is this test asking?* If I ask for cached data that doesn't exist yet, does
the code return `None` instead of crashing?

It asks for a series called `NONEXISTENT_SERIES` and checks that the return value
is `None`.

*Passing means:* the cache layer handles a miss gracefully. The calling code can
then decide what to do (fetch from API, load from sample, raise an error) without
the cache layer itself blowing up.

*Industry relevance:* Graceful failure modes are what separate code that survives
production from code that works in demos. A `None` return is explicit and
handleable. An unexpected crash at 2am is not.

---

### Validation

---

**`test_validate_series_rejects_negative_unemployment`**

*What is this test asking?* If the unemployment rate data contains a negative
value, does the code catch it and refuse to proceed?

Unemployment cannot be negative by definition — it's a percentage of people
looking for work. The test creates a fake UNRATE series with a -0.1 value buried
in the middle and checks that `validate_series` raises a `DataValidationError`.

*Passing means:* the validator catches the impossible value and raises an error
with "negative" in the message.

*Industry relevance:* Data quality is the unglamorous foundation of every model.
A negative unemployment rate in your training data will confuse any model that
sees it. Catching it at ingestion — before it ever reaches a model — is the right
place to fail.

---

**`test_validate_series_rejects_excess_nans`**

*What is this test asking?* If a series has a long run of missing values, does
the code refuse to use it?

It builds a 100-observation series with 8 consecutive `NaN` values starting at
position 10. Eight consecutive missing observations is 8% of the series — above
the 5% consecutive threshold the project enforces. The test checks that this
triggers a `DataValidationError` mentioning "consecutive NaN."

*Passing means:* large gaps in the data are caught before they silently distort
a model. A single missing month is annoying. Eight in a row is a data problem
that needs investigation, not interpolation.

*Industry relevance:* In economic data, long NaN runs often signal a reporting
gap or a series that was discontinued. Proceeding blindly fills those gaps with
an assumption (usually linear interpolation), which can introduce phantom trends
that don't exist in the real world.

---

**`test_validate_series_passes_clean_data`**

*What is this test asking?* Does a clean, well-behaved series pass validation
without triggering any errors?

It creates a smooth unemployment series that increases linearly from 4% to 6%
over 20 months — no negatives, no gaps — and checks that `validate_series`
returns it unchanged.

*Passing means:* the validator doesn't over-flag. Good data gets through.

*Industry relevance:* Overly strict validation that rejects valid data is its
own failure mode. If your pipeline treats normal data as suspicious, it will
constantly halt on data that's perfectly fine to use.

---

### Regime labeling

---

**`test_regime_labeling_recession_flag`**

*What is this test asking?* Are dates we know were inside a recession correctly
labeled as "recession"?

It creates a series of three monthly observations from January 2009 — well
inside the 2007–2009 Global Financial Crisis recession per NBER dates — and
checks that all three are labeled "recession."

*Passing means:* the hardcoded NBER recession dates in the code correctly match
the known historical record. January 2009 should be recession. It is.

*Industry relevance:* Regime-aware modeling is a live research area. Knowing how
to label and condition on macroeconomic regimes is a real skill in applied
econometrics and quant research — whether you're at a central bank, an asset
manager, or a policy shop.

---

**`test_regime_labeling_expansion_flag`**

*What is this test asking?* Are dates we know were in an expansion correctly
NOT labeled as recession?

It builds a 30-month series from January 2003 — between the 2001 recession and
the 2007 recession, a clear expansion — and checks that observations from 2005
are not labeled "recession."

*Passing means:* the labeling logic correctly leaves expansion periods alone.
No false positives.

*Industry relevance:* A model that treats 2005 as a crisis period will learn
completely wrong relationships. Getting the regime label right is upstream of
everything else — wrong labels mean wrong regime-conditional results throughout
the entire benchmark.

---

**`test_regime_labeling_high_volatility`**

*What is this test asking?* When a series suddenly becomes extremely noisy, does
the code correctly flag that stretch as "high volatility"?

It creates 60 observations of a perfectly stable series (all 5.0) followed by
20 observations with massive random swings (standard deviation 50x higher than
normal). The test checks that at least some of the noisy observations get labeled
"high_volatility."

*Passing means:* the rolling standard deviation check is working. It doesn't just
look at NBER dates — it also catches volatility spikes that NBER doesn't
officially classify as recessions.

*Industry relevance:* Volatility spikes matter as much as recessions for coverage
analysis. The 2010–2011 European debt crisis, the 2015 China slowdown, and the
2020 commodity crash were all periods of elevated volatility that weren't fully
captured by NBER recession windows. This test makes sure those periods are labeled
separately.

---

### Data fetching

---

**`test_fetch_series_uses_cache`**

*What is this test asking?* When fresh cached data exists, does the code use it
without making an API call to FRED?

It creates a fake cache file, patches the code to report that cache as fresh,
and checks that the function `fetch_series` returns data without ever calling
the FRED API. If the API were called, the test would fail.

*Passing means:* the priority order works — cache first, API only when necessary.

*Industry relevance:* FRED rate-limits API calls and requires a key. Any
production pipeline that hits the API every time it runs will eventually break
or get throttled. Caching with staleness detection is the standard pattern, and
verifying it works is non-negotiable.

---

**`test_fetch_series_fallback_no_key`**

*What is this test asking?* If there's no API key at all, does the code fall back
to pre-saved sample data instead of crashing?

It patches out the API key (returns `None`) and directs the code to a temporary
directory pre-populated with small sample files. It checks that the function
returns non-empty data with no negative unemployment values.

*Passing means:* CI pipelines that run without a FRED key (like a public GitHub
Actions runner) still work. The project degrades gracefully instead of throwing a
credentials error.

*Industry relevance:* Secrets management is a real engineering concern. Any code
that assumes credentials are always available will fail in CI, in other
developers' environments, and in any deployment that doesn't have the key
configured. A working fallback path is how you make code portable.

---

## `tests/test_models.py` — 13 tests

These tests check that all five forecasting models behave correctly: they train
on data, produce forecasts of the right shape, and correctly signal when they
can or can't do something.

---

### Base class conformance

---

**`test_all_models_implement_base`**

*What is this test asking?* Do all four currently-implemented models
(ARIMA, ETS, Random Forest, XGBoost) actually inherit from the shared
`ForecastModel` base class?

It instantiates one of each model and checks `isinstance(model, ForecastModel)`
for all of them. No fitting, no prediction — just "does this class play by the
rules of the interface?"

*Passing means:* every model can be treated interchangeably by the conformal
wrapper, which only knows about `ForecastModel` and doesn't care which specific
model it's wrapping.

*Industry relevance:* In any production system that swaps models in and out —
backtesting frameworks, model registries, A/B testing infrastructure — this
interface contract is what makes pluggability possible without rewriting the
plumbing.

---

### ARIMA

---

**`test_arima_fit_predict`**

*What is this test asking?* Does ARIMA fit on training data and return a forecast
of the correct shape?

It trains ARIMA on 100 observations of synthetic data and asks for a 6-step
forecast. It checks that the output is a 6-element array with no `NaN` values.

*Passing means:* ARIMA runs to completion without crashing and returns six finite
numbers, one for each forecast step.

*Industry relevance:* ARIMA is the baseline every other model is compared against
in academic and applied forecasting work. If you can't show that ARIMA's
conformal intervals under-cover during recessions while your method doesn't, the
entire project's thesis falls apart. It has to work correctly first.

---

**`test_arima_gaussian_intervals`**

*What is this test asking?* Do the Gaussian (parametric) confidence intervals
from ARIMA have the right shape, and are the lower bounds actually below the
upper bounds?

It trains ARIMA, requests 6-step intervals at 90% confidence, and checks three
things: lower has 6 values, upper has 6 values, and `upper > lower` for every
step.

*Passing means:* ARIMA produces a proper 90% interval, not an inverted one or a
degenerate zero-width one. This is the Gaussian interval we'll later compare
against the conformal interval.

*Industry relevance:* Understanding what Gaussian intervals are and why they fail
on fat-tailed economic data is foundational to the entire case for conformal
prediction. ARIMA's Gaussian intervals assume normally distributed errors. Recessions
violate that assumption systematically, which is the empirical story this project tells.

---

**`test_arima_name`**

*What is this test asking?* Does `ARIMAModel().name` return the string `"ARIMA"`?

*Passing means:* the model identifies itself correctly. This string shows up in
benchmark result tables, plot legends, and dashboard labels.

*Industry relevance:* Metadata discipline — every model knowing its own name and
reporting it consistently — is what makes benchmark results reproducible and
readable. It sounds trivial until you have twelve model variants and a results
table with unlabeled rows.

---

### ETS

---

**`test_ets_fit_predict`**

*What is this test asking?* Does ETS fit on training data and return a 6-step
forecast with no missing values?

Same structure as the ARIMA test. Trains on 100 synthetic observations, asks
for 6 steps, checks shape and no `NaN`.

*Passing means:* ETS runs cleanly. The model selects one of its five internal
configurations (additive vs. multiplicative error and trend, with or without
damping) and produces a complete forecast.

*Industry relevance:* ETS is widely used in supply chain, economic forecasting,
and central bank nowcasting. It's in R's `forecast` package and Python's
`statsmodels`. Understanding it is a practical skill, not just an academic one.

---

**`test_ets_gaussian_intervals`**

*What is this test asking?* Do ETS's prediction intervals have the right shape
and correct direction?

ETS generates its intervals through simulation — it runs 500 hypothetical futures
forward from the fitted model and reads off the quantiles. The test checks that
this simulation produces valid bounds: 6 values each, upper above lower.

*Passing means:* the simulation-based interval is working. No numerical failures,
no inverted bounds.

*Industry relevance:* Simulation-based intervals are the standard when closed-form
expressions aren't available. Understanding when and why a model uses simulation
vs. an analytical formula is the kind of practical judgment that separates someone
who has read the statsmodels docs from someone who understands what's happening
underneath.

---

**`test_ets_name`**

*What is this test asking?* Does `ETSModel().name` return `"ETS"`?

*Passing means:* the model labels itself correctly in results.

*Industry relevance:* Same reasoning as ARIMA — consistent model identity is table
stakes for any system that stores or displays benchmark results.

---

### Random Forest

---

**`test_rf_fit_predict`**

*What is this test asking?* Does Random Forest fit on lag features built from
the training series and return a 6-step recursive forecast?

It trains the model on 100 observations, which internally get converted into
a feature matrix of 12 lags, rolling means, and rolling standard deviations.
The test checks that the output is a 6-element array with no `NaN`.

*Passing means:* the feature engineering pipeline works and the recursive
multi-step prediction (where each predicted value becomes the next step's
input) doesn't produce garbage.

*Industry relevance:* Feature engineering for time series — knowing which
lags matter, when rolling statistics are appropriate, how to handle the
recursion for multi-step forecasts — is a core applied ML skill. Random Forest
on engineered features is competitive with much fancier approaches on many
real economic series.

---

**`test_rf_gaussian_raises`**

*What is this test asking?* When you ask a fitted Random Forest for Gaussian
prediction intervals, does it explicitly refuse with a `NotImplementedError`?

It trains the model and calls `predict_gaussian()`, expecting an error.

*Passing means:* Random Forest correctly signals that it has no parametric
interval. Any code that asks for a Gaussian interval from this model will get
a clear error, not a silent failure or a made-up number.

*Industry relevance:* Non-parametric models don't produce parametric intervals.
This is a design decision, not a limitation — these models are deliberately
conformal-only in this project. Understanding why a model raises here (and what
to use instead) is the kind of thing that comes up in any applied ML interview
at a quant shop or tech company working with forecasting.

---

**`test_rf_name`**

*What is this test asking?* Does `RandomForestModel().name` return `"RandomForest"`?

*Passing means:* consistent model identity across the benchmark.

*Industry relevance:* See above. Consistent naming is what makes output readable.

---

### XGBoost

---

**`test_xgb_fit_predict`**

*What is this test asking?* Does XGBoost fit and produce a valid 6-step forecast?

Same structure as the Random Forest test — same feature engineering, same recursive
prediction logic, just with XGBoost's boosted trees instead of a random forest.
Checks shape and no `NaN`.

*Passing means:* XGBoost trains in reasonable time on 100 observations and
produces six finite forecast values.

*Industry relevance:* XGBoost has been the default choice for structured data
in industry for a decade. Knowing how to wrap it for time series — including
the feature engineering and recursion required — is a practical skill that
shows up constantly in applied econometrics and data science roles.

---

**`test_xgb_gaussian_raises`**

*What is this test asking?* Does XGBoost also correctly refuse to produce Gaussian
intervals?

Same as the Random Forest case. XGBoost is a non-parametric model. Gaussian
intervals don't apply. The test checks that the code says so clearly.

*Passing means:* explicit refusal with `NotImplementedError`. No silent wrongs.

*Industry relevance:* XGBoost's uncertainty quantification is an active research
area — quantile regression, conformal prediction, Bayesian wrappers. Understanding
that the base model has no native interval, and that you need to add one explicitly,
is foundational to any serious treatment of ML uncertainty in practice.

---

**`test_xgb_name`**

*What is this test asking?* Does `XGBoostModel().name` return `"XGBoost"`?

*Passing means:* consistent model identity in benchmark output.

*Industry relevance:* Same as the others. Boring but necessary.

---

## `tests/test_conformal.py` — 36 tests

This is the core of Week 2. These tests cover the rolling calibration splitter,
the six evaluation metrics, the conformal wrapper that adds prediction intervals
to any model, and the LSTM model itself.

Conformal prediction is the methodological heart of this project. The idea: instead
of assuming a distribution for your forecast errors (as Gaussian intervals do), you
learn the error distribution directly from held-out data — the calibration set —
and use that to set your interval width. If you do it correctly, the resulting
intervals have a provable coverage guarantee regardless of what distribution your
errors actually follow.

---

### Splitter

The splitter is what enforces honest calibration. It takes a series of `n`
observations and divides them into three non-overlapping, temporally ordered
groups: training (fit the model), calibration (measure its errors), and test
(evaluate coverage). The invariant throughout: the calibration set is always
from the future relative to training, and the test set is always from the future
relative to calibration.

---

**`test_rolling_split_temporal_order`**

*What is this test asking?* Is the last training index strictly before the first
calibration index, and the last calibration index strictly before the first test
index?

It runs `rolling_calibration_split` on a 150-observation series and checks the
boundary conditions directly.

*Passing means:* the split produces a true temporal chain: training → calibration
→ test, with no gaps or inversions.

*Industry relevance:* This is the single most important structural property of
time series validation. Any model evaluated with shuffled or randomly assigned
train/test splits on time series data is leaking future information into the
training process. The results look good and are meaningless. Getting this right
is what separates serious time series work from cargo-cult ML.

---

**`test_rolling_split_sizes_sum_to_n`**

*What is this test asking?* Do the three groups together account for every
observation in the series — no observations double-counted, none left out?

It checks `len(train) + len(cal) + len(test) == n`.

*Passing means:* the split is a true partition. Every observation goes to exactly
one group.

*Industry relevance:* Missing or duplicated observations in train/eval splits can
produce subtly wrong coverage estimates. Verifying completeness is basic hygiene
that prevents a class of quiet errors in evaluation pipelines.

---

**`test_rolling_split_no_overlap`**

*What is this test asking?* Do train and cal share any indices? Do cal and test
share any indices?

It checks `train ∩ cal == ∅` and `cal ∩ test == ∅`.

*Passing means:* the sets are disjoint. No observation is used for both fitting
and evaluating.

*Industry relevance:* Data leakage — using the same observations to both train a
model and evaluate it — is one of the most common and consequential mistakes in
applied ML. This test directly guards against it.

---

**`test_rolling_split_min_train_floor`**

*What is this test asking?* When the series is short enough that the fractional
training size would fall below 100 observations, does the code floor it at 100?

It uses a 160-observation series with a 60% training fraction (which would give
96 observations) and checks that the training set still has 100 observations.

*Passing means:* the `max(100, int(n * frac))` logic is working. No model gets
trained on fewer than 100 observations regardless of the series length.

*Industry relevance:* Most time series models need a minimum amount of data to
produce stable parameter estimates. ARIMA with 20 observations is not the same
model as ARIMA with 100. Enforcing a floor is a real design choice that prevents
degenerate results on short series.

---

**`test_rolling_split_too_short_raises`**

*What is this test asking?* If the series is so short that train and calibration
together use up all the observations with nothing left for testing, does the code
raise an error rather than silently producing an empty test set?

It passes a 110-observation series to `rolling_calibration_split`, where training
alone requires 100 and calibration requires 22 — together more than 110.

*Passing means:* a clear `ValueError` with "too short" in the message. The code
refuses to proceed rather than producing nonsense.

*Industry relevance:* Silent failures in ML pipelines are the hardest bugs to
find. A function that returns an empty array instead of raising when given bad
input will cascade through the rest of the system in ways that are extremely hard
to trace. Loud, informative failures are the correct design.

---

**`test_rolling_calibration_splitter_temporal_order`**

*What is this test asking?* For every window in the rolling re-calibration
schedule, does training end before calibration ends, and does calibration end
before the batch ends?

`RollingCalibrationSplitter` generates multiple (train_end, cal_end, batch_end)
triples for a 200-observation series and checks the ordering holds for all of them.

*Passing means:* the rolling windows respect time order even as they advance
through the series.

*Industry relevance:* Rolling window evaluation — where you re-train and re-calibrate
as new data arrives — is how you simulate real deployment. Brokerages, central
banks, and forecasting desks all use this structure. Verifying the temporal
ordering holds across all windows is the proof that the simulation is honest.

---

**`test_rolling_calibration_splitter_batch_end_advances`**

*What is this test asking?* As the rolling windows move forward through the series,
does each successive batch end at a strictly later point in time than the previous?

It checks that the list of `batch_end` values is strictly sorted ascending and
has no duplicates.

*Passing means:* the rolling schedule moves forward monotonically. Each evaluation
window covers new ground.

*Industry relevance:* A rolling evaluator that revisits the same time periods
twice, or that stops advancing, would produce duplicated or incomplete results.
This test catches both failure modes.

---

### Evaluation metrics

These tests verify the six scoring functions used to judge how well the prediction
intervals actually perform. Each function takes the true observed values and the
predicted intervals and returns a number summarizing performance.

---

**`test_empirical_coverage_perfect`**

*What is this test asking?* When every true value falls inside its prediction
interval, does `empirical_coverage` return exactly 1.0?

It creates three observations (1.0, 2.0, 3.0) with intervals that each fully
contain the observation, and checks the result.

*Passing means:* 100% coverage is correctly computed as 1.0.

*Industry relevance:* Coverage is the fundamental metric for prediction intervals.
If your model says "90% confidence interval," it should cover the true value 90%
of the time. Verifying the formula works correctly on known cases is the minimum
bar for trusting the evaluation.

---

**`test_empirical_coverage_zero`**

*What is this test asking?* When no true value falls inside its interval, does
coverage return 0.0?

The observations are (10, 20, 30) and the intervals are all [0, 1] — nowhere
close.

*Passing means:* zero coverage is correctly computed as 0.0.

*Industry relevance:* The zero-coverage case is a useful sanity check. An
evaluator that returns 0.5 when nothing is covered has a bug, and that bug
would make every interval look better than it is.

---

**`test_empirical_coverage_partial`**

*What is this test asking?* When exactly half of the observations are covered,
does coverage return 0.5?

Two observations, one inside the interval and one outside.

*Passing means:* partial coverage computes correctly, not just the extremes.

*Industry relevance:* Most real intervals fall somewhere between perfect and
zero coverage. Verifying that the interpolation is correct is what makes the
metric trustworthy across the whole range.

---

**`test_mean_interval_width`**

*What is this test asking?* Does `mean_interval_width` correctly average the
widths of two intervals?

Intervals of width 2 and width 4 should average to 3.

*Passing means:* the formula is `mean(upper - lower)`, computed correctly.

*Industry relevance:* Width matters as much as coverage. A prediction interval
that says "GDP growth next year will be somewhere between -50% and +50%" has
perfect coverage and is completely useless. Narrow intervals that still cover
are the goal, and width is how you measure the cost.

---

**`test_winkler_score_no_misses`**

*What is this test asking?* When all observations are covered, does the Winkler
score equal the mean interval width?

The Winkler score is width plus a penalty for any miss. If there are no misses,
the penalty term is zero, and the score should equal the width exactly.

*Passing means:* no-miss case produces the correct result.

*Industry relevance:* The Winkler score is the standard combined metric for
interval forecasts — it penalizes both wide intervals and misses in a single
number. It's the interval forecast equivalent of RMSE for point forecasts. Any
serious interval evaluation uses it.

---

**`test_winkler_score_miss_penalty`**

*What is this test asking?* When an observation falls outside the interval by
some amount δ, is the penalty correctly computed as (2/α) × δ?

With α = 0.10 (targeting 90% coverage), a miss by 1.0 unit should add a penalty
of 20.0 on top of the interval width.

*Passing means:* the penalty formula is correct. Misses are charged at the right
rate.

*Industry relevance:* The 2/α multiplier is what makes the Winkler score
sensitive to the chosen confidence level. At 90% confidence, a miss costs 20×
the distance. At 95%, it costs 40×. Understanding this scaling is how you
interpret Winkler scores across different alpha values without confusing yourself.

---

**`test_picp_matches_empirical_coverage`**

*What is this test asking?* Does `picp()` (Prediction Interval Coverage
Probability) return the same value as `empirical_coverage()` for the same inputs?

They're the same calculation under two different names from two different
literature traditions. The test just confirms they're both calling the same code.

*Passing means:* the two names are consistent.

*Industry relevance:* PICP comes from the neural network forecasting literature;
empirical coverage comes from statistics. They mean the same thing. Knowing that
both exist and are equivalent is the kind of cross-domain literacy that's useful
when reading papers from different communities.

---

**`test_mpiw_normalized`**

*What is this test asking?* Does `mpiw()` (Mean Prediction Interval Width) divide
the mean width by the provided standard deviation?

With mean width 3 and `y_std = 2`, the result should be 1.5.

*Passing means:* normalization is correct.

*Industry relevance:* Raw interval widths aren't comparable across series with
different scales — the width of an unemployment rate interval (measured in
percentage points) can't be directly compared to the width of an oil price
interval (measured in dollars per barrel). Normalizing by series volatility
makes comparisons meaningful.

---

**`test_mpiw_invalid_std_raises`**

*What is this test asking?* If you pass a standard deviation of zero (or negative),
does `mpiw` raise a `ValueError` instead of dividing by zero?

*Passing means:* the function catches the invalid input and fails loudly.

*Industry relevance:* Division by zero in a metric function would silently produce
`inf` or `NaN`, which then propagates silently through the benchmark results. An
explicit `ValueError` catches the degenerate case immediately.

---

**`test_coverage_deviation_zero`**

*What is this test asking?* When empirical coverage exactly equals the target
(1 − α), does `coverage_deviation` return 0?

*Passing means:* the deviation formula `|empirical − target|` correctly returns
zero when there's no gap.

*Industry relevance:* Coverage deviation is the headline metric of this project.
It's the single number that answers "how wrong is your interval?" A conformal
interval should be within ±3% of target in all regimes. A Gaussian interval
during a recession often misses by 15–20%. This is the formula that tells you
which is which.

---

**`test_coverage_deviation_nonzero`**

*What is this test asking?* When empirical coverage is 0% against a 90% target,
does `coverage_deviation` return 0.90?

Observations of 10.0 with intervals [0, 1] — nothing is covered — against a
target of 90% coverage means the deviation is exactly 0.90.

*Passing means:* the deviation is computed correctly at the worst-case extreme.

*Industry relevance:* The worst-case behavior of your evaluation metrics matters
as much as the typical case. If the formula breaks at 0% coverage, you can't
trust what it says when coverage is 72% — which is the kind of number you actually
see for Gaussian ARIMA during recessions.

---

**`test_evaluate_all_returns_all_keys`**

*What is this test asking?* Does `evaluate_all()` — the function that computes
all six metrics in one call — return a dictionary containing all six expected
keys?

It checks for: `coverage`, `width`, `winkler`, `picp`, `mpiw`, `coverage_deviation`.

*Passing means:* the benchmark runner (which calls `evaluate_all` per model, per
series, per regime) will always have all six metrics to work with.

*Industry relevance:* Consistent output schema from evaluation functions is what
makes benchmarks reproducible. If the function returns five keys one day and six
the next, downstream analysis breaks silently.

---

**`test_evaluate_all_consistent`**

*What is this test asking?* Do the values in `evaluate_all`'s output match what
you'd get calling each metric function individually?

It runs `evaluate_all` and then runs `empirical_coverage`, `mean_interval_width`,
and `winkler_score` separately on the same inputs, checking that the numbers match.

*Passing means:* `evaluate_all` is a faithful wrapper. It doesn't do any
additional transformation or rounding that would change the values.

*Industry relevance:* A convenience function that quietly transforms its inputs
before computing metrics would produce results that can't be reproduced by calling
the underlying functions directly. This kind of silent divergence is a real source
of benchmark inconsistencies in applied research.

---

### ConformalWrapper

The `ConformalWrapper` class takes any model and adds honest prediction intervals
to it. It does this by measuring the model's errors on a held-out calibration set
and using those errors to set the interval width. The tests below verify that the
wrapper works correctly for every model, produces valid intervals, and fails loudly
when misused.

---

**`test_conformal_wrapper_calibrate_sets_q_hat`**

*What is this test asking?* After calling `calibrate()` on a real series, is the
internal conformal quantile (`_q_hat`) set to a positive number?

It wraps ARIMA with `ConformalWrapper`, calibrates on a 150-observation series,
and checks that the quantile is not `None` and is greater than zero.

*Passing means:* calibration ran to completion and produced a meaningful error
threshold.

*Industry relevance:* The conformal quantile is the core output of calibration —
it's the margin you add to point forecasts to get intervals. If it's zero, your
intervals have zero width. If it's `None`, nothing was calibrated. Verifying it's
set and positive is the first sanity check on any conformal implementation.

---

**`test_conformal_wrapper_interval_shape`**

*What is this test asking?* When you ask for a 6-step forecast interval, do you
get two arrays of length 6?

It calls `predict_interval(6)` and checks that both `lower` and `upper` are shape
`(6,)`.

*Passing means:* the interval matches the requested horizon. One lower bound and
one upper bound per forecast step.

*Industry relevance:* Mismatched array shapes — getting a 5-step interval for a
6-step forecast request, or a scalar when you expected an array — cause downstream
crashes that are confusing to debug. Shape verification is the simplest form of
output contract checking.

---

**`test_conformal_wrapper_lower_lt_upper`**

*What is this test asking?* Is the lower bound strictly below the upper bound for
every forecast step?

*Passing means:* the intervals are valid. No inverted bounds, no zero-width
intervals.

*Industry relevance:* An inverted interval (`lower > upper`) is a degenerate
result that signals a numerical failure somewhere in the calibration. An interval
scoring function applied to inverted bounds produces garbage. Checking this
explicitly is how you catch that failure mode before it propagates.

---

**`test_conformal_wrapper_no_nan`**

*What is this test asking?* Do the interval bounds contain any `NaN` values?

It uses ETS (which uses simulation internally) and checks that neither `lower`
nor `upper` contains any missing values after calibration and prediction.

*Passing means:* no numerical failures in the calibration or prediction pipeline.

*Industry relevance:* `NaN` propagation is one of the most common silent failure
modes in numerical code. A single `NaN` in a model's prediction contaminates the
entire interval, and downstream code that doesn't check for it will produce wrong
results quietly.

---

**`test_conformal_wrapper_requires_calibrate`**

*What is this test asking?* If you call `predict_interval()` without first calling
`calibrate()`, does the code raise a `RuntimeError` instead of returning a wrong
answer?

It creates a wrapper, skips calibration, and calls `predict_interval` directly,
expecting an error mentioning "calibrate."

*Passing means:* misuse produces a clear, actionable error message. The code
refuses to produce an uncalibrated interval.

*Industry relevance:* A function that silently returns zero-width intervals or
`None` when called out of order is far more dangerous than one that raises. If
you're comparing conformal intervals across models and one was accidentally never
calibrated, you'd have a subtle bug that would be nearly impossible to spot in
the results table.

---

**`test_conformal_wrapper_all_models[ARIMAModel]`**
**`test_conformal_wrapper_all_models[ETSModel]`**
**`test_conformal_wrapper_all_models[RandomForestModel]`**
**`test_conformal_wrapper_all_models[XGBoostModel]`**

*What is this test asking?* Does the conformal wrapper work end-to-end with each
of the four non-LSTM models?

This is one parametrized test that runs four times — once per model class. Each
run calibrates the wrapper on the 150-observation synthetic series and checks that
`predict_interval(6)` returns valid bounds (correct shape, upper above lower).

*Passing means:* the wrapper is genuinely model-agnostic. It works through the
`ForecastModel` interface without caring about the specific implementation.

*Industry relevance:* A wrapper that only works with one model type is not a
wrapper — it's a special case. True model-agnosticism is what makes the conformal
infrastructure reusable across the entire benchmark. This is also a real software
engineering pattern: write your evaluation layer against an interface, not an
implementation.

---

### LSTM

---

**`test_lstm_implements_base`**

*What is this test asking?* Does `LSTMModel` inherit from `ForecastModel` and
pass the `isinstance` check?

*Passing means:* LSTM plugs into the same interface as every other model. The
conformal wrapper, the benchmark runner, and any future code written against
`ForecastModel` will all work with LSTM without modification.

*Industry relevance:* Deep learning models are often treated as special cases
requiring their own infrastructure. One of the design goals here is to show they
don't have to be — if you wrap them in a clean interface, they're just another
forecaster. That's a useful mental model in any applied ML context.

---

**`test_lstm_name`**

*What is this test asking?* Does `LSTMModel().name` return `"LSTM"`?

*Passing means:* the LSTM identifies itself correctly in benchmark output.

*Industry relevance:* Same as all the other name tests. Consistent model identity
in results and plots is what makes the benchmark readable.

---

**`test_lstm_fit_predict_shape`**

*What is this test asking?* Does the LSTM train on an 80-observation series with
a 12-step lookback window and return a 6-step forecast with no `NaN`?

It uses `max_epochs=3` (instead of the default 200) to keep the test fast. Three
epochs is not enough to fully train the network, but it is enough to verify that
the training loop runs without crashing, the weights update, and the prediction
pipeline returns finite numbers.

*Passing means:* the LSTM training and prediction pipeline runs to completion on
real data.

*Industry relevance:* Training stability is the first thing you verify for any
neural network before worrying about accuracy. A network that crashes on the first
forward pass or produces `NaN` gradients has a bug that no amount of hyperparameter
tuning will fix.

---

**`test_lstm_predict_without_fit_raises`**

*What is this test asking?* If you call `predict()` on a fresh LSTM without
fitting it first, does it raise a `RuntimeError`?

*Passing means:* uninitialized prediction fails loudly with a clear message.

*Industry relevance:* An uninitialized neural network's weights are random. A
`predict()` call on a freshly instantiated model would return random noise —
which would look like a valid forecast to any downstream code that doesn't check
for this condition. The explicit error is what prevents that.

---

**`test_lstm_gaussian_raises`**

*What is this test asking?* Does `predict_gaussian()` on a fitted LSTM raise
`NotImplementedError`?

The LSTM has no parametric error distribution. Its prediction intervals come from
the conformal wrapper, not from the model itself. This test confirms that the
model correctly refuses to fake one.

*Passing means:* the LSTM signals clearly that Gaussian intervals don't apply to it.

*Industry relevance:* Neural networks' uncertainty quantification is an active
and contested research area. Bayesian deep learning, MC Dropout, deep ensembles,
conformal prediction — there are many approaches, and none of them reduce to a
simple Gaussian interval. Knowing what a model can and can't produce, and having
that represented explicitly in the interface, is practical software design.

---

**`test_lstm_with_conformal_wrapper`**

*What is this test asking?* Does the LSTM work end-to-end inside a conformal
wrapper — calibration, quantile computation, and interval prediction — and
produce valid bounds?

It uses the 150-observation series (not the short 80-observation series, because
the wrapper needs at least 100 observations for the training floor). Trains for
3 epochs for speed. Checks that the resulting intervals have the right shape and
that lower is below upper.

*Passing means:* the full pipeline works: LSTM as a base model, conformal wrapper
adding valid intervals on top.

*Industry relevance:* Combining deep learning with conformal prediction is a
genuinely active research direction. The idea that you can take any black-box
model — neural network, gradient boosted tree, even a language model — and add
provably valid coverage guarantees on top is one of the more practically useful
ideas in recent ML theory. This test is the end-to-end proof that the combination
works.

---

**`test_lstm_too_short_raises`**

*What is this test asking?* If you try to fit the LSTM on only 10 observations
when it needs a 24-step lookback window, does it raise a `ValueError` with a
clear message?

You can't build a 24-step input sequence from 10 observations. The test confirms
the code detects this and refuses to proceed.

*Passing means:* the LSTM fails informatively on data that's too short to be
useful.

*Industry relevance:* In practice, economic series vary widely in length. Monthly
unemployment goes back to 1948; some niche financial series have 18 months of
history. Any model that silently produces garbage on short series, instead of
raising, will contaminate results in a benchmark that mixes series of different
lengths. The explicit check is the right design.

---

*This project currently has 61 tests across three files. The plan is to add
`tests/test_benchmark.py` in Week 3 when the benchmark runner is implemented.*
