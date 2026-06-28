# 60-Era Benchmark Report

This page is the benchmark-oriented companion to the lighter sample artifacts in `docs/sample_artifacts/`. It is meant for the stronger walk-forward setup based on `configs/strong_benchmark_catboost.yaml`.

## Scope

- objective: broader offline benchmark over a larger walk-forward window
- config: `configs/strong_benchmark_catboost.yaml`
- validation design: `15` folds x `4` validation eras = `60` validation eras
- main model branch: LightGBM + CatBoost
- MLX status: experimental branch, tracked separately and not part of the main benchmark claim

## Setup

```bash
cd /Users/nandhuraj/Repository/Numerai
uv sync --extra dev --extra catboost
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export MKL_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
nice -n 10 uv run python scripts/backtest_walkforward.py --config configs/strong_benchmark_catboost.yaml
```

Optional holdout-aware reblend:

```bash
uv run python scripts/reblend_walkforward.py \
  --config configs/strong_benchmark_catboost.yaml \
  --source-run artifacts/<source_run_dir> \
  --optimize-weights \
  --objective mean_corr \
  --grid-step 0.05 \
  --holdout-folds 3
```

## Config Summary

| Item | Value |
| --- | --- |
| train era limit | `180` |
| minimum train eras | `24` |
| validation eras per fold | `4` |
| embargo eras | `1` |
| max folds | `15` |
| total validation eras | `60` |
| base model mix | `lgbm_main`, `lgbm_alt`, `catboost_optional` |

## Model Leaderboard

Populate this section from the generated `model_leaderboard.csv` after the benchmark run completes.

Suggested command:

```bash
LATEST=$(ls -td artifacts/* | head -1)
cat "$LATEST/model_leaderboard.csv"
```

## Holdout Table

When `--holdout-folds` is used during reblend optimization, the run writes `holdout_evaluation.csv` with split-level metrics:

- `selection_folds`
- `holdout_folds`
- `all_folds`

The table includes:

- mean correlation
- correlation standard deviation
- Sharpe-like score
- max drawdown
- mean feature exposure
- max feature exposure
- number of eras

## Plots

After the benchmark run completes, attach or link:

- `plots/cumulative_corr.png`
- `plots/model_comparison.png`
- `plots/era_corr_histogram.png`

## Interpretation Notes

- This remains an offline backtest, not live trading evidence.
- The stronger 60-era setup is more credible than the 12-era sample run for model comparison.
- Weight optimization should be interpreted more cautiously when no untouched holdout folds are reserved.
- Neutralization is included as a diagnostic and post-processing experiment, not as a guaranteed improvement.

## Current Status

This page is prepared for the stronger benchmark path. Replace the placeholders above with concrete artifacts once the `strong_benchmark_catboost.yaml` run and optional holdout-aware reblend are finished.
