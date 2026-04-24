from importlib import import_module
import json
from pathlib import Path
from typing import Iterable, List


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from splitters import BaseSplitter
from models import BaseModel, ModelConfig
from benchmark import AnalystModel, EvaluateModel


DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_TEST_SIZE = 0.3
DEFAULT_DATA_DIR = Path("../data/raw")
DEFAULT_SPLIT_DIR = Path("../data/splitted")
DEFAULT_RESULTS_DIR = Path("Results")
DEFAULT_CONFIG_DIR = Path("config")


def _load_datasets(data_dir: Path = DEFAULT_DATA_DIR, dataset_names: List[str] = None):
    datasets = []
    dataset_filter = set(dataset_names) if dataset_names is not None else None

    for parquet_path in sorted(data_dir.glob("*/*.parquet")):
        dataset_name = parquet_path.stem
        if dataset_filter is not None and dataset_name not in dataset_filter:
            continue
        datasets.append((dataset_name, pd.read_parquet(parquet_path)))

    if dataset_filter is not None:
        found_names = {name for name, _ in datasets}
        missing = sorted(dataset_filter - found_names)
        if missing:
            raise ValueError(f"Datasets not found in {data_dir}: {missing}")

    if not datasets:
        raise ValueError(f"No parquet datasets found in {data_dir}")

    return datasets


def _instantiate_objects(
    classes: Iterable[type],
    seeds: List[int] = None,
    keep_size: bool = False,
):
    active_seeds = seeds if seeds is not None else DEFAULT_SEEDS
    return [
        _class(seeds = active_seeds, keep_size = keep_size)
        for _class in classes
    ]


def _string2class(
    modules: List[object] = None,
    inputs: List[str] = None,
    conversion_engine: str = "splitters",
):
    """
    This procedure convert list of strings to either implemented Splitters or Models

    Parameters:
        modules: List[object], None as default. The extra modules imported which implement additional partitioning regimes and models
        inputs: List[str], None as default. The list of splitter/model names
        conversion_engine: str, "splitters" as default. Manages what types of class will be converted to. Valid arguments are: "splitters" and "models".
    Returns:
        List[BaseSplitter] or List[BaseModel]
    """

    def _class_extract_from_module():
        if modules is None:
            raise ValueError("modules must be provided when converting strings to classes")
        if inputs is None:
            raise ValueError("inputs must be provided when converting strings to classes")

        result_classes = []
        for input_str in inputs:
            module_found = False
            for module in modules:
                module_obj = module
                if isinstance(module, str):
                    try:
                        module_obj = import_module(f"{conversion_engine}.{module}")
                    except ModuleNotFoundError:
                        continue

                if hasattr(module_obj, input_str):
                    class_ = getattr(module_obj, input_str)
                    result_classes.append(class_)
                    module_found = True
                    break

            if not module_found:
                raise ValueError(f"Class {input_str} not found in any of the provided modules.")

        return result_classes

    if conversion_engine in ["splitters", "models"]:
        return _class_extract_from_module()

    raise ValueError("Valid conversion engines are 'splitters' or 'models'")


# ================================================ Pipeline procedures ================================================

def main_split(
    splitters: List[BaseSplitter],
    dataset_names: List[str] = None,
    test_size: float = DEFAULT_TEST_SIZE,
    data_dir: Path = DEFAULT_DATA_DIR,
):
    datasets = _load_datasets(data_dir = data_dir, dataset_names = dataset_names)

    for dataset_name, df in datasets:
        print(f"Processing dataset: {dataset_name}")
        for splitter in splitters:
            print(f"  Running splitter: {splitter.__class__.__name__}")
            splitter.split(file_name = dataset_name, df = df.copy(deep=True), test_size = test_size)


def main_train(
    models: List[type],
    dataset_names: List[str] = None,
    config_dir: Path = DEFAULT_CONFIG_DIR,
):
    Evaluator = EvaluateModel(
        models,
        dataset_names,
        DEFAULT_SPLIT_DIR,
        DEFAULT_RESULTS_DIR,
        config_dir_location = config_dir,
    )
    Evaluator.evaluate()


def main_eval(
    splitwise_baseline_only: bool = True,
    splitwise_include_variants: bool = False,
    modelwise_eval: bool = False,
    per_dataset_table_eval: bool = False,
):
    Analyst = AnalystModel(results_root = DEFAULT_RESULTS_DIR, split_data_root = DEFAULT_SPLIT_DIR)

    print("============================================================")
    print("Running split-agnostic evaluation on baseline models")
    Analyst.split_agnostic_test()

    print("============================================================")
    print("Running split-wise evaluation")
    Analyst.split_wise_test(
        baseline_only = splitwise_baseline_only,
        include_variants = splitwise_include_variants,
    )

    if modelwise_eval:
        print("============================================================")
        print("Running model-wise evaluation")
        Analyst.model_wise_test(
            baseline_only = splitwise_baseline_only,
            include_variants = splitwise_include_variants,
        )

        print("============================================================")
        print("Running model-wise diagnostic (Random Split as benchmark)")
        Analyst.model_wise_vs_random_latex_table(
            baseline_only = splitwise_baseline_only,
            include_variants = splitwise_include_variants,
            baseline_split = "Random_Split"
        )

        print("============================================================")
        print("Running model-wise evaluation (Random Split as benchmark)")
        Analyst.robustness_model_comparison_latex(
            baseline_only = splitwise_baseline_only,
            include_variants = splitwise_include_variants,
        )

    if per_dataset_table_eval:
        print("============================================================")
        print("Running per-dataset full table export")
        Analyst.per_dataset_table_test(
            baseline_only = splitwise_baseline_only,
            include_variants = splitwise_include_variants,
            print_latex = True,
        )




def main(
    modules: List[object] = None,
    splitters: List[str] = None,
    models: List[str] = None,
    require_eval: bool = False,
    splitwise_baseline_only: bool = True,
    splitwise_include_variants: bool = False,
    modelwise_eval: bool = False,
    per_dataset_table_eval: bool = False,
    dataset_names: List[str] = None,
    seeds: List[int] = None,
    test_size: float = DEFAULT_TEST_SIZE,
    keep_size: bool = False,
    data_dir: Path = DEFAULT_DATA_DIR,
    config_dir: Path = DEFAULT_CONFIG_DIR,
):
    """
    This main() procedure presents the primary pipeline of this OOD Toolkit:
        1. Splitting processes
        2. Training and Testing models on Splitted datasets
        3. Constructing performance tables and statistic-based comparisons
    Parameters:
        modules: List[object], None as default. The extra modules imported which implement additional partitioning regimes and models
        splitters: List[str], None as default. The list of partitioning techniques
        models: List[str], None as default. The list of robustness-tested models
        require_eval: bool, False as default. Specify if statistic-based comparisons and performance tables are required
        dataset_names: List[str], None as default. The list of benchmark dataset names
        seeds: List[int], None as default. The list of seeds for reproducibility
        keep_size: bool False as default. Set to True to keep the big-sized data, >1M samples
        data_dir: Path, "../data/raw" as default. Location of folder saving benchmark datasets
        config_dir: Path, "config/" as default. Location of folder saving runtime and models' hyperparameters configurations
        splitwise_baseline_only: bool, True as default. Compare only baseline models in split-wise statistical tests
        splitwise_include_variants: bool, False as default. Include model variants as separate competitors in split-wise tests
        modelwise_eval: bool, False as default. Run model-wise statistical tests that compare split types for each model
        per_dataset_table_eval: bool, False as default. Print one full nRMSE table per dataset (models x splits), plus LaTeX rows
    Note:
        The split-agnostic table only uses the baseline configuration. In contrast, the split-wise tables are more flexible.
            To construct a full split-wise table including all model variants set splitwise_baseline_only = False and splitwise_include_variants = True.

            To construct a split-wise table only including baseline models, set splitwise_baseline_only = True, the same table will be presented for whatever the splitwise_include_variants is set, as splitwise_baseline_only is prioritised over splitwise_include_variants.
    """

    if modules is not None:
        print("The extra modules imported:")
        for module in modules:
            print(module if isinstance(module, str) else module.__name__)
        print("============ End of modules printing ============")
    else:
        print("No new modules are passed. Apply the existing modules")

    if splitters is not None:
        splitter_classes = _string2class(
            modules = modules,
            inputs = splitters,
            conversion_engine = "splitters",
        )
        splitter_instances = _instantiate_objects(
            splitter_classes,
            seeds = seeds,
            keep_size = keep_size,
        )
        main_split(
            splitter_instances,
            dataset_names = dataset_names,
            test_size = test_size,
            data_dir = data_dir,
        )
    else:
        print("Partitioning mechanism is not activated")

    # Training and Testing
    if models is not None:
        model_classes = _string2class(
            modules = modules,
            inputs = models,
            conversion_engine = "models",
        )
        main_train(model_classes, dataset_names = dataset_names, config_dir = config_dir)
    else:
        print("No models are provided")

    if require_eval:
        main_eval(
            splitwise_baseline_only = splitwise_baseline_only,
            splitwise_include_variants = splitwise_include_variants,
            modelwise_eval = modelwise_eval,
            per_dataset_table_eval = per_dataset_table_eval,
        )


if __name__ == "__main__":
    split_modules = ["geometric_split", "marginal_distribution_shift", "random_split"]
    model_modules = ["statistical_models", "tree_models", "resnet"]
    modules = split_modules + model_modules
    splitters = None # ["BasicGeometricSplit", "RandomSplit", "MarginalDistributionSplit"]
    models = [
        # "HuberLinearRegressor", "HuberPolynomialRegressor", "KNNRegressor", "SVMRegressor",
        # "DTRegressor", "RFRegressor", "GBRegressor", "ABRegressor", "XGBRegressor", "LightGBMRegressor",
        # "ResnetRegressor",
    ]
    dataset_names = ["synthetic_0"]
    main(
        modules,
        splitters,
        models,
        require_eval = False,
        splitwise_baseline_only = False,
        splitwise_include_variants = False,
        modelwise_eval = True,
        per_dataset_table_eval = False,
        dataset_names = dataset_names
    )
