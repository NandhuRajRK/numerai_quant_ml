# Colab Workflow

Use local smoke tests for pipeline robustness, then use Colab for the heavy walk-forward research run.

## Recommended Split

- Local MacBook:
  - `uv run python scripts/download_data.py`
  - `uv run python scripts/backtest_walkforward.py --config configs/local_smoke.yaml`
- Colab:
  - full `configs/baseline.yaml` walk-forward backtest
  - optional XGBoost-enabled runs
  - repeated experiment sweeps

## Why This Split Works

- The local smoke config proves the pipeline, checkpoints, and artifact writing work.
- The full baseline run is CPU-heavy and can be uncomfortable on a laptop for repeated iteration.
- Colab is more useful for experimentation volume than for tiny debugging loops.

## Practical Colab Steps

1. Clone the repo into Colab.
2. Install system dependency for LightGBM if needed by the runtime.
3. Install Python dependencies.
4. Upload or mount the Numerai data directory, or redownload there.
5. Run the full backtest with `configs/baseline.yaml`.

## Suggested Commands

```bash
git clone <your-repo-url>
cd numerai-quant-ml
python -m pip install uv
uv sync --extra dev
uv run python scripts/download_data.py
uv run python scripts/backtest_walkforward.py
```

If you want XGBoost there too:

```bash
uv sync --extra dev --extra xgboost
```

## What To Check During A Long Run

- `artifacts/<run_name>/status.json`
- `artifacts/<run_name>/fold_metrics.partial.csv`
- `artifacts/<run_name>/model_leaderboard.partial.csv`
- `artifacts/<run_name>/checkpoints/`

Those files update before the final report is written, so interrupted runs still leave useful evidence.
