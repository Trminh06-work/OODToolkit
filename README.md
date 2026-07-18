# OODToolkit: An end-to-end OOD Tabular Regression Assessment Pipeline

OODToolkit is a small research toolkit for benchmarking regression models under out-of-distribution (OOD) train/test splits. It lets you:

- generate multiple split regimes from tabular datasets,
- train several classical and neural regression models on those splits,
- compare performance across split types and model variants.

The repository already includes benchmark datasets under `data/raw/` (each with a `Data_Statistics_Summary/` folder), one prepared split tree under `data/splitted/bike/`, model variant configs under `src/config/`, and sample outputs under `src/Results/`.

## What The Pipeline Does

The main workflow in [`src/main.py`](src/main.py) has three stages:

1. **Split datasets** into in-distribution and OOD-style train/test partitions, optionally injecting Gaussian noise into the training targets.
2. **Train models** on each saved split and write per-dataset JSON results, including wall-clock training and inference times.
3. **Evaluate results** with aggregate tables (plain-text and LaTeX) and statistical comparisons: split-agnostic, split-wise, model-wise, per-dataset, robustness vs `Random_Split`, and runtime tables.

This is a script-driven repo rather than a packaged CLI. By default, the relative paths in `src/main.py` assume you run commands from the `src/` directory.

## Repository Layout

```text
OODToolkit/
├── data/
│   ├── raw/                 # Input datasets, one folder per dataset (with Data_Statistics_Summary/)
│   └── splitted/            # Generated split files
├── logs/                    # Slurm job logs (created by batch runs)
├── script/
│   ├── bash.sh              # Slurm batch entrypoint
│   └── job.conf             # Batch job configuration
├── src/
│   ├── benchmark/           # Training evaluation and statistical analysis
│   ├── config/              # Per-model variant JSON configs
│   ├── models/              # Regression model implementations
│   ├── splitters/           # OOD and random split generators
│   ├── Results/             # Example saved outputs
│   └── main.py              # Main pipeline entrypoint
└── requirements.txt
```

## Requirements

- Python 3.10+ is recommended.
- Install dependencies from [`requirements.txt`](requirements.txt).
- Datasets must be stored as Parquet files.

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Extra dependencies for specific models (not in `requirements.txt`):

- `xgboost`, `lightgbm` for `XGBRegressor` / `LightGBMRegressor` (included in `requirements.txt`)
- `tabicl` for `TabiclRegressor`
- `pytabkit` for `RealMLPRegressor`

Install these only if you plan to run the corresponding models:

```bash
pip install tabicl pytabkit
```

## Data Format

Each dataset is expected under:

```text
data/raw/<dataset_name>/<dataset_name>.parquet
```

The code assumes:

- the file is a tabular Parquet dataset,
- all feature columns come first,
- the **last column is the regression target**.

Example existing dataset:

```text
data/raw/bike/bike.parquet
```

The repo ships with ~28 real-world regression datasets under `data/raw/` (e.g. `bike`, `diamonds`, `protein`, `house_sales`, `kin40k`, `3droad`). Each dataset folder also contains a `Data_Statistics_Summary/` folder with a Markdown summary of the dataset's statistics.

The default `dataset_names` in the `__main__` block of `src/main.py` reference `synthetic_*` datasets that are **not** included in the repo — replace them with the dataset names you actually have before running `python main.py` directly.

## Implemented Splitters

Available splitter classes in `src/splitters/`:

- `RandomSplit` (module `random_split`) → `Random_Split`
- `BasicGeometricSplit` (module `geometric_split`) → `Single_Hyperball`, `Multiple_Hyperballs`, `Single_Slab`, `Semi_Infinite_Slab`, `KMeans_Hyperballs`, plus `Reverse_*` counterparts of each when `include_reverse=True` (the default)
- `MarginalDistributionSplit` (module `marginal_distribution_shift`) → `Covariate_Shift`

Saved outputs follow this pattern:

```text
data/splitted/<dataset_name>/<split_name>/train_<i>.parquet
data/splitted/<dataset_name>/<split_name>/test_<i>.parquet
```

with one `train_<i>`/`test_<i>` pair per seed (default seeds: 42–46).

**Noise injection:** splitters can add zero-mean Gaussian noise (sigma = 1/16) to the **training targets only** via `add_noise_to_train`. Note that `main()` currently hard-codes `add_noise_to_train=True` when it runs the splitting stage; call `main_split(...)` directly if you want noise-free training splits.

## Implemented Models

Available model classes in `src/models/`, grouped by module (the module name is what you pass in `modules=[...]`):

- `statistical_models`: `HuberLinearRegressor`, `HuberPolynomialRegressor`, `KNNRegressor`, `SVMRegressor`
- `tree_models`: `DTRegressor`, `RFRegressor`, `GBRegressor`, `ABRegressor`, `XGBRegressor`, `LightGBMRegressor`
- `resnet`: `ResnetRegressor` (deep learning, inherits `BaseDLModel`)
- `slip_interpolant`: `SLipInterpolant` (smooth Lipschitz interpolation, backed by the GPU/MPS-enabled `liblipt.py`)
- `mlp`: `BaselineMLPRegressor` (scikit-learn MLP), `RealMLPRegressor` (requires `pytabkit`), `TabiclRegressor` (requires `tabicl`)

Model variants are defined in JSON files under [`src/config/`](src/config). If a model has a matching config file, each named variant is trained separately. Results are written to:

```text
src/Results/<ModelName>/<variant_name>/
```

Each variant folder includes:

- `<dataset_name>.json` with metrics by split type,
- `_variant.json` with the exact config that was used.

## Quick Start

### Option 1: Use Existing Split Files

If you want a fast first run, reuse the prepared `bike` splits and only train a model.

From the repository root:

```bash
cd src
python
```

Then run:

```python
from main import main

main(
    modules=["tree_models"],
    splitters=None,
    models=["RFRegressor"],
    require_eval=True,
    splitwise_baseline_only=True,
    modelwise_eval=False,
    dataset_names=["bike"],
)
```

This will:

- read split files from `../data/splitted/bike/`,
- train `RFRegressor` variants defined in `config/RFRegressor.json`,
- write results to `src/Results/RFRegressor/`,
- run the evaluation stage.

### Option 2: Generate New Splits Then Train

```bash
cd src
python
```

```python
from main import main

main(
    modules=[
        "geometric_split",
        "marginal_distribution_shift",
        "random_split",
        "tree_models",
    ],
    splitters=["RandomSplit", "BasicGeometricSplit", "MarginalDistributionSplit"],
    models=["RFRegressor"],
    require_eval=True,
    modelwise_eval=False,
    dataset_names=["bike"],
)
```

This runs the full pipeline for the `bike` dataset.

Additional evaluation flags on `main()`:

- `splitwise_include_variants=True` (with `splitwise_baseline_only=False`): include model variants as separate competitors in split-wise tests
- `modelwise_eval=True`: compare split types for each model, including diagnostics against `Random_Split` and a robustness (relative performance degradation) table
- `per_dataset_table_eval=True`: print one full nRMSE table (models x splits) per dataset
- `runtime_eval=True`: print a training/inference runtime comparison table

The evaluation stage requires `dataset_names` to be specified explicitly.

You can also run the script directly with `python main.py` from `src/` — edit the `__main__` block at the bottom of [`src/main.py`](src/main.py) to select modules, splitters, models, and datasets first.

## Running On Slurm

The repo includes a Slurm wrapper at [`script/bash.sh`](script/bash.sh) and a default config at [`script/job.conf`](script/job.conf).

Typical usage:

```bash
sbatch script/bash.sh --config script/job.conf
```

Key config fields:

- `RUN_MODE`: `pipeline` (default) runs the split/train/eval phases; `visualize` runs a visualization-only phase (note: `visualize` mode expects a `main_visualize()` function that is not currently present in `src/main.py`)
- `MODULES`: module files to import from `src/splitters` and `src/models`
- `SPLITTERS`: splitter class names to run (leave empty to skip splitting)
- `MODELS`: model class names to train
- `ARRAY_MODELS` / `ARRAY_DATASETS`: optional Slurm-array sharding — each array task runs one model and/or one dataset; when both are set the array iterates their cartesian product (array size must equal `n_datasets * n_models`)
- `REQUIRE_EVAL`: whether to run the evaluation stage
- `SPLITWISE_BASELINE_ONLY` / `SPLITWISE_INCLUDE_VARIANTS`: control which model variants enter the split-wise tests
- `MODELWISE_EVAL`: whether to run model-wise tests across split types for each model
- `PER_DATASET_TABLE_EVAL`: print one full nRMSE table per dataset (models x splits)
- `RUNTIME_EVAL`: print a runtime performance table (training/inference time per model)
- `DATASET_NAMES`: comma-separated dataset names such as `bike` (leave empty to use every dataset under `data/raw/`)
- `VISUALIZE_*`: grid size, plot kinds, split names, and run ids for `RUN_MODE="visualize"`
- `PYTHON_BIN` / `CONDA_ENV_NAME`: Python executable and conda environment used inside the job

Slurm job logs (stdout/stderr, including the printed evaluation tables) are written to `logs/` at the repository root.

## Understanding Outputs

After a run, the main output locations are:

- `data/splitted/`: saved train/test Parquet files for each split regime
- `src/Results/<Model>/<variant>/`: metrics for each dataset
- `src/Results/.../_variant.json`: exact runtime/model parameters used

The metric JSON files store split-level results for each run, including:

- `MSE`, `RMSE`, `MAE`, `MaxAE`
- `nRMSE`, `nMAE`, `nMaxAE` (normalized by the training-target standard deviation)
- `Adjusted R2 score`
- `MAPE`, `sMAPE`
- `training_time`, `inference_time` (wall-clock seconds)

During training, only the **feature space** is standardized (`StandardScaler`); the target is left unnormalized. The evaluation stage uses `nRMSE` as its key comparison metric and can emit LaTeX-formatted tables (mean ranks with Friedman/Holm post-hoc tests, model-wise vs `Random_Split` diagnostics, robustness degradation, and runtime comparisons).

## Customizing Model Variants

To change hyperparameters, edit or add JSON files in [`src/config/`](src/config). The filename must match the model class name.

Example:

```text
src/config/RFRegressor.json
```

Structure:

```json
{
  "variants": {
    "baseline": {
      "runtime_config": {
        "seed": 42
      },
      "model_params": {
        "n_estimators": 400
      }
    }
  }
}
```

`runtime_config` maps to `ModelConfig`, and `model_params` are passed directly to the model constructor.

## Adding Your Own Models And Splitters

You can extend the toolkit by adding your own model and splitter implementations.

### Add A New Model

Place the implementation in [`src/models/`](src/models).

Requirements:

- tabular models should inherit from `BaseModel`
- deep learning models should inherit from `BaseDLModel`
- the class must implement the expected training/prediction interface used by the toolkit
- the model class name is what you pass in `models=[...]`

In practice this means:

- accept `df_train`, `df_test`, and `config` in the constructor
- call the parent constructor
- implement `fit()`
- implement `predict()`

If you want the model to support named hyperparameter variants, add a matching JSON file in [`src/config/`](src/config). The filename must match the class name, for example:

```text
src/models/my_models.py
src/config/MyCustomRegressor.json
```

To make the model discoverable, include its module name in `modules=[...]` and its class name in `models=[...]`.

Example:

```python
main(
    modules=["tree_models", "my_models"],
    models=["RFRegressor", "MyCustomRegressor"],
)
```

### Add A New Splitter

Place the implementation in [`src/splitters/`](src/splitters).

Requirements:

- inherit from `BaseSplitter`
- implement `split(...)` in the same style as the existing splitters
- save outputs under `data/splitted/<dataset_name>/<split_name>/`
- write paired `train_<i>.parquet` and `test_<i>.parquet` files

The splitter must follow the repository convention that:

- features come from all columns except the last one
- the last column is the target
- each saved train/test file contains both features and target

To use the splitter, include its module name in `modules=[...]` and its class name in `splitters=[...]`.

Example:

```python
main(
    modules=["random_split", "my_splitters"],
    splitters=["RandomSplit", "MyCustomSplit"],
)
```

If you want your custom classes to be importable from package-level imports, also update [`src/models/__init__.py`](src/models/__init__.py) or [`src/splitters/__init__.py`](src/splitters/__init__.py), although the dynamic module loading in `main.py` primarily depends on the module names you pass in `modules=[...]`.

## Notes For First-Time Use

- Run from `src/` unless you explicitly override the default paths in `main.py`.
- Start with `dataset_names=["bike"]` to keep the first run small.
- If you only want to test training, leave `splitters=None` and reuse `data/splitted/bike/`.
- Large datasets (>1M rows) are downsampled to 800K rows by splitters unless `keep_size=True`.
- Splitting through `main()` always injects Gaussian noise into the training targets; use `main_split(...)` directly for noise-free splits.
- The prepared `data/splitted/bike/` tree contains the seven non-reverse split types; regenerating splits with `BasicGeometricSplit` will also produce the `Reverse_*` variants.
- For those conducting experiments via Slurm, the final results are available in the `OODToolkit/logs/` folder.

## Where To Look Next

- Entry point: [`src/main.py`](src/main.py)
- Model configs: [`src/config/README.md`](src/config/README.md)
- Batch runner: [`script/bash.sh`](script/bash.sh)

## ✉️ Contact

Author: `Bao Minh Tran`

GitHub: [@Trminh06-work](https://github.com/Trminh06-work)

LinkedIn: [Bao Minh Tran](https://www.linkedin.com/in/bao-minh-tran-587272372)

Email:
- Deakin's student mail: s224236373@deakin.edu.au (May be expired after 2027)
- General mail: trminh06.work@gmail.com

Feel free to open an issue if you have questions, suggestions, or find a bug.
