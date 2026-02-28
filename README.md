# codes_e_new

Predict steel-slag carbonation efficiency with multi-model training and an optimized ensemble (OE).

## Quick Start

1) Enter the project directory:

`cd codes_e_new`

2) Run a full experiment (creates `runs/` and a new run directory under it):

`python main.py`

3) Quick sanity run (small number of epochs, used to verify the pipeline works end-to-end):

`python main.py --quick`

## Data

By default, the project uses `data/The sample data used in the study.xlsx`.

## Outputs (a new `run_id` is created under `runs/` for each run)

`runs/<run_id>/`
- `config.json`: Full configuration used in this run
- `models/`: Best checkpoints for each model
- `excel/`: All exported tables used for re-plotting and analysis (metrics, predictions, training history, ensemble mechanism, etc.)
- `figures/`: Evaluation figures (loss, R², scatter plots, residual diagnostics, metric comparisons, etc.)

## Key Principle: Avoid Data Leakage

- `train/val` are used for training and early stopping (model selection and tuning)
- `test` is only used for final generalization evaluation and analysis, and must not be used for tuning
