# OODToolkit: An end-to-end OOD Tabular Regression Assessment Pipeline

OODToolkit is a small research toolkit for benchmarking regression models under out-of-distribution (OOD) train/test splits. It lets you:

- generate multiple split regimes from tabular datasets,
- train several classical and neural regression models on those splits,
- compare performance across split types and model variants.

The repository already includes example datasets under `data/raw/`, one prepared split tree under `data/splitted/bike/`, model variant configs under `src/config/`, and sample outputs under `src/Results/`.

## What The Pipeline Does

The main workflow in [`src/main.py`](src/main.py) has three stages:

1. **Split datasets** into in-distribution and OOD-style train/test partitions.
2. **Train models** on each saved split and write per-dataset JSON results.
3. **Evaluate results** with aggregate tables and statistical comparisons.

This is a script-driven repo rather than a packaged CLI. By default, the relative paths in `src/main.py` assume you run commands from the `src/` directory.

## Repository Layout

```text
OODToolkit/
├── data/
│   ├── raw/                 # Input datasets, one folder per dataset
│   └── splitted/            # Generated split files
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

## Implemented Splitters

Available splitter classes in `src/splitters/`:

- `RandomSplit`
- `BasicGeometricSplit`
- `MarginalDistributionSplit`

These generate directories such as:

- `Random_Split`
- `Single_Hyperball`
- `Multiple_Hyperballs`
- `Single_Slab`
- `Semi_Infinite_Slab`
- `KMeans_Hyperballs`
- `Covariate_Shift`

Saved outputs follow this pattern:

```text
data/splitted/<dataset_name>/<split_name>/train_<i>.parquet
data/splitted/<dataset_name>/<split_name>/test_<i>.parquet
```

## Implemented Models

Available model classes in `src/models/`:

- `HuberLinearRegressor`
- `HuberPolynomialRegressor`
- `KNNRegressor`
- `SVMRegressor`
- `DTRegressor`
- `RFRegressor`
- `GBRegressor`
- `ABRegressor`
- `XGBRegressor`
- `LightGBMRegressor`
- `ResnetRegressor`

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

## Running On Slurm

The repo includes a Slurm wrapper at [`script/bash.sh`](script/bash.sh) and a default config at [`script/job.conf`](script/job.conf).

Typical usage:

```bash
sbatch script/bash.sh --config script/job.conf
```

Key config fields:

- `MODULES`: module files to import from `src/splitters` and `src/models`
- `SPLITTERS`: splitter class names to run
- `MODELS`: model class names to train
- `REQUIRE_EVAL`: whether to run the evaluation stage
- `MODELWISE_EVAL`: whether to run model-wise tests across split types for each model
- `DATASET_NAMES`: comma-separated dataset names such as `bike`

## Understanding Outputs

After a run, the main output locations are:

- `data/splitted/`: saved train/test Parquet files for each split regime
- `src/Results/<Model>/<variant>/`: metrics for each dataset
- `src/Results/.../_variant.json`: exact runtime/model parameters used

The metric JSON files store split-level results for metrics including:

- `MSE`
- `RMSE`
- `MAE`
- `Adjusted R2 score`
- `MAPE`
- `sMAPE`

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
- Large datasets may be downsampled by splitters unless `keep_size=True`.
- For those conducting experiments via Slurm, the final results are available in the OODToolkit/logs/ folder.

## Where To Look Next

- Entry point: [`src/main.py`](src/main.py)
- Model configs: [`src/config/README.md`](src/config/README.md)
- Batch runner: [`script/bash.sh`](script/bash.sh)

## ✉️ Contact

Author: `Bao Minh Tran`

GitHub: [@Trminh06-work](https://github.com/Trminh06-work)

LinkedIn: [Bao Minh Tran](www.linkedin.com/in/bao-minh-tran-587272372)

Email:
- Deakin's student mail: s224236373@deakin.edu.au (May be expired after 2027)
- General mail: trminh06.work@gmail.com

Feel free to open an issue if you have questions, suggestions, or find a bug.
