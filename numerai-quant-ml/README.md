# numerai-quant-ml

Portfolio-grade Numerai Tournament research and live prediction pipeline. This repo is designed to show quant ML judgment and ML engineering discipline: time-aware validation, multi-model ensembling, Numerai-specific neutralization, reproducible run artifacts, and a safe submission workflow.

This project is for learning, experimentation, and portfolio use only. It does not include staking logic, does not ship real secrets, does not claim profitability, and is not trading advice.

## What Numerai Is

Numerai is a hedge-fund-backed data science tournament where participants train models on obfuscated market features and submit live predictions. You never see raw tickers or traditional price series. That makes it a strong sandbox for practicing noisy supervised learning, temporal validation, model stability analysis, and live ML operations.

## What Makes This Repo Stronger Than a Baseline

- Walk-forward era validation instead of one static split.
- Multi-model ensemble with configurable weights.
- Optional XGBoost support through a `uv` extra.
- Feature-neutralized post-processing to reduce raw feature exposure.
- Run artifact bundles with metrics, plots, leaderboards, predictions, configs, and reports.
- Stable saved ensemble bundle for live prediction generation.
- Dry-run-first submission flow that refuses to run without environment credentials.

## Repository Structure

```text
configs/                  Project and model configuration
data/                     Ignored raw data and prediction outputs
models/                   Ignored saved baseline and ensemble bundles
artifacts/                Ignored run-by-run research outputs
reports/                  Optional generated reports
notebooks/                Lightweight EDA and inspection notebook
scripts/                  CLI entry points
src/numerai_quant/        Reusable package code
tests/                    Unit tests for metrics, format, and validation logic
.github/workflows/        CI for lint and tests
```

## Setup With uv

```bash
cd numerai-quant-ml
uv sync --extra dev
cp .env.example .env
```

To enable XGBoost in the advanced model zoo:

```bash
uv sync --extra dev --extra xgboost
```

Then set `advanced.model_zoo[].enabled: true` for the XGBoost entry in `configs/baseline.yaml`.

## Core Commands

Download Numerai train, validation, and live data:

```bash
uv run python scripts/download_data.py
```

Train the original LightGBM baseline:

```bash
uv run python scripts/train_baseline.py
```

Validate the original baseline on the validation parquet:

```bash
uv run python scripts/validate_baseline.py
```

Run the advanced walk-forward backtest across the model zoo:

```bash
uv run python scripts/backtest_walkforward.py
```

Train the final ensemble bundle for live use:

```bash
uv run python scripts/train_ensemble.py
```

Generate live predictions from the advanced ensemble bundle:

```bash
uv run python scripts/predict_live.py
```

Run the full portfolio workflow in one command:

```bash
uv run python scripts/run_portfolio_pipeline.py
```

Dry-run submit a prediction file:

```bash
uv run python scripts/submit_predictions.py data/predictions/live_predictions_<round>.csv
```

Actually upload predictions:

```bash
uv run python scripts/submit_predictions.py data/predictions/live_predictions_<round>.csv --submit
```

Run quality checks:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```

## Validation Philosophy

Numerai data is organized into eras, which act like time slices. This repo treats time structure seriously.

The advanced research workflow:

- builds walk-forward folds with a minimum training window;
- applies an era embargo before each validation block;
- evaluates each model and the ensemble on unseen future eras;
- summarizes stability with mean CORR, standard deviation, Sharpe-like score, max drawdown, and feature exposure.

This is not the same as live performance, but it is much more credible than a random split or a single static validation pass.

## Ensemble and Neutralization

The advanced pipeline trains a configurable model zoo from `configs/baseline.yaml`, blends raw predictions with explicit weights, then optionally neutralizes the ensemble against the most exposed features.

This does not pretend to fully solve MMC or guarantee better tournament performance. The point is to show you understand how to build a stronger signal stack and how to talk about Numerai-specific tradeoffs in a technically honest way.

## Artifacts

Each advanced backtest creates an ignored artifact run directory containing:

- `summary.json`
- `fold_metrics.csv`
- `model_leaderboard.csv`
- `oof_predictions.parquet`
- `feature_exposure.csv`
- `plots/cumulative_corr.png`
- `plots/era_corr_histogram.png`
- `plots/model_comparison.png`
- `report.md`

Final ensemble fitting also snapshots the saved production bundle into a run directory for traceability.

## CORR and MMC

CORR is the rank-style correlation between predictions and Numerai targets. This repo directly optimizes for a CORR-like modeling setup and evaluates per-era Spearman correlation as a practical proxy.

MMC measures how much unique signal a model contributes beyond Numerai's meta-model. This repo does not fully optimize MMC, but the ensemble and neutralization layers make the project much more relevant to that discussion than a plain baseline.

## Environment Variables

The submission script expects:

- `NUMERAI_PUBLIC_ID`
- `NUMERAI_SECRET_KEY`
- `NUMERAI_MODEL_ID`

They belong in `.env`. The script validates their presence and never prints the secret values.

## Assumptions

- Dataset download uses the common Numerai `numerapi.download_dataset("<version>/<file>")` pattern.
- The default dataset version remains `v5.2` until you update `configs/baseline.yaml`.
- The advanced workflow uses the train parquet for walk-forward backtesting and can optionally append validation rows when fitting the final live ensemble.
- XGBoost support is optional and disabled by default so the repo stays runnable from a fresh clone without extra setup.

## Limitations

- Walk-forward validation is stronger than a single split, but it is still backtesting.
- Neutralization here is intentionally simple and not a full Numerai research stack.
- No staking, capital allocation, or trading layer is included.
- Live tracking and model registry integration are left as next steps.

## Next Up

- Add richer target support if Numerai exposes multiple trainable targets in the chosen dataset version.
- Add parameter sweeps with artifact-level experiment comparison.
- Add live round tracking and historical submission monitoring.
- Add feature clustering or model diversification diagnostics.
