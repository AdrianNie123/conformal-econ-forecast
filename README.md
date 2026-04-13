# Conformal Forecast

> When Gaussian intervals lie, conformal prediction tells the truth.

<!-- Full README written in Week 5. See PRD.md Section 8 for spec. -->

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env  # add your FRED API key
python -m conformal_econ.data.fred --refresh
pytest tests/
```

## Status

Week 1 — data pipeline and model foundation in progress.
