from __future__ import annotations

import os
import pandas as pd
import numpy as np

import json
import gc
from pathlib import Path
from typing import List

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

from sklearn.preprocessing import StandardScaler

from models import BaseModel, ModelConfig

import logging

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message=r".*tensorboardX.*removed.*",
    category=UserWarning,
    module=r"pytorch_lightning.*",
)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)



class Evaluator:
    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true


    def score_MSE(self):
        mse = mean_squared_error(self.y_true, self.y_pred)
        return mse


    def score_RMSE(self):
        rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
        return rmse


    def score_MAE(self):
        mae = mean_absolute_error(self.y_true, self.y_pred)
        return mae


    def score_r2(self, use_adjusted = False, num_feat = None):
        r2 = r2_score(self.y_true, self.y_pred)
        if use_adjusted:
            if num_feat is None:
                print("missing number of features")
                return None
            n = len(self.y_true)
            r2 = 1 - ((1 - r2) * ((n - 1) / (n - num_feat - 1)))
        return r2


    def score_MAPE(self):
        mape = mean_absolute_percentage_error(self.y_true, self.y_pred)
        return mape


    def score_sMAPE(self):
        y_true = np.asarray(self.y_true, dtype=float).ravel()
        y_pred = np.asarray(self.y_pred, dtype=float).ravel()

        epsilon = 1e-4 # avoid zero division
        denom = np.abs(y_true) + np.abs(y_pred) + epsilon
        num = np.abs(y_true - y_pred)
        sMAPE = 200.0 * np.mean(num / denom)

        return sMAPE


    def score_nRMSE(self):
        rmse = self.score_RMSE()
        return rmse / self.y_true.std()


    def score_nMAE(self):
        mae = self.score_MAE()
        return mae / self.y_true.std()



class EvaluateModel:
    def __init__(self,
        models: List[BaseModel],
        dataset_names: List[str] = None,
        source_dir_location = None,
        result_dir_location = None,
        config = None,
        config_dir_location = None,
    ):
        self.models = models
        self.dataset_names = dataset_names
        self.source_dir = Path(source_dir_location) if source_dir_location is not None else None
        self.target_dir = Path(result_dir_location) if result_dir_location is not None else None
        self.config = ModelConfig() if config is None else config
        self.config_dir = Path(config_dir_location) if config_dir_location is not None else None


    def _default_variant(self):
        return [{
            "name": "default",
            "runtime_config": {},
            "model_params": {},
        }]


    def _sanitize_variant_name(self, variant_name: str) -> str:
        sanitized = str(variant_name).strip().replace(os.sep, "_").replace("/", "_")
        return sanitized or "default"


    def _load_model_variants(self, model_name: str):
        if self.config_dir is None:
            return self._default_variant()

        config_file = self.config_dir / f"{model_name}.json"
        if not config_file.exists():
            return self._default_variant()

        with config_file.open("r", encoding = "utf-8") as f:
            raw_config = json.load(f)

        raw_variants = raw_config.get("variants", raw_config) if isinstance(raw_config, dict) else raw_config
        variants = []

        if isinstance(raw_variants, dict):
            for variant_name, variant_payload in raw_variants.items():
                if not isinstance(variant_payload, dict):
                    raise ValueError(f"Variant '{variant_name}' in {config_file} must be an object")
                variants.append({
                    "name": variant_name,
                    "runtime_config": dict(variant_payload.get("runtime_config", {})),
                    "model_params": dict(variant_payload.get("model_params", {})),
                })
        elif isinstance(raw_variants, list):
            for variant_payload in raw_variants:
                if not isinstance(variant_payload, dict):
                    raise ValueError(f"Every variant entry in {config_file} must be an object")
                if "name" not in variant_payload:
                    raise ValueError(f"Every variant entry in {config_file} must include a 'name'")
                variants.append({
                    "name": variant_payload["name"],
                    "runtime_config": dict(variant_payload.get("runtime_config", {})),
                    "model_params": dict(variant_payload.get("model_params", {})),
                })
        else:
            raise ValueError(f"Unsupported config format in {config_file}")

        if not variants:
            raise ValueError(f"No variants defined in {config_file}")

        return variants


    def evaluate(self):
        def _sanitize(obj):
            if isinstance(obj, dict):
                return {str(k): _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, tuple):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if hasattr(obj, "item"):
                return obj.item()
            raise TypeError

        if self.source_dir is None:
            raise ValueError("source_dir_location must be provided")
        if self.target_dir is None:
            raise ValueError("result_dir_location must be provided")
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory does not exist: {self.source_dir}")
        if not self.source_dir.is_dir():
            raise NotADirectoryError(f"Source path is not a directory: {self.source_dir}")
        if self.config_dir is not None:
            if not self.config_dir.exists():
                raise FileNotFoundError(f"Config directory does not exist: {self.config_dir}")
            if not self.config_dir.is_dir():
                raise NotADirectoryError(f"Config path is not a directory: {self.config_dir}")

        self.target_dir.mkdir(parents = True, exist_ok = True)

        for model_class in self.models:
            model_name = model_class.__name__
            print(f"Training model: {model_name}")
            model_variants = self._load_model_variants(model_name)

            dataset_dirs = sorted(path for path in self.source_dir.iterdir() if path.is_dir())
            if not dataset_dirs:
                raise ValueError(f"No split datasets found in {self.source_dir}")
            if self.dataset_names is not None:
                dataset_filter = set(self.dataset_names)
                found_dataset_names = {path.name for path in dataset_dirs}
                missing_datasets = sorted(dataset_filter - found_dataset_names)
                if missing_datasets:
                    raise ValueError(f"Datasets not found in {self.source_dir}: {missing_datasets}")
                dataset_dirs = [path for path in dataset_dirs if path.name in dataset_filter]

            for variant in model_variants:
                variant_name = self._sanitize_variant_name(variant["name"])
                base_runtime_config = dict(self.config.__dict__)
                base_runtime_config.update(variant["runtime_config"])
                variant_runtime_config = ModelConfig(**base_runtime_config)
                variant_model_params = dict(variant["model_params"])

                model_output_dir = self.target_dir / model_name / variant_name
                model_output_dir.mkdir(parents = True, exist_ok = True)

                variant_metadata_file = model_output_dir / "_variant.json"
                with variant_metadata_file.open("w", encoding = "utf-8") as f:
                    json.dump(variant, f, indent = 2, default = _sanitize)

                print(f"  Variant: {variant_name}")

                for dataset_dir in dataset_dirs:
                    dataset_name = dataset_dir.name
                    results = {}
                    print(f"    Dataset: {dataset_name}")

                    for split_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
                        split_name = split_dir.name
                        train_files = sorted(split_dir.glob("train_*.parquet"))

                        if not train_files:
                            continue

                        split_results = {}

                        for train_file in train_files:
                            if not train_file.stem.startswith("train_"):
                                print(f"      Skip file {train_file.name}: unexpected train file format")
                                continue

                            idx = train_file.stem.removeprefix("train_")
                            if not idx:
                                print(f"      Skip file {train_file.name}: missing run identifier")
                                continue

                            test_file = split_dir / f"test_{idx}.parquet"
                            if not test_file.exists():
                                print(f"      Skip run {idx}: missing {test_file.name}")
                                continue

                            try:
                                df_train = pd.read_parquet(train_file)
                                df_test = pd.read_parquet(test_file)

                                X_train = df_train.iloc[:, :-1]
                                y_train = df_train.iloc[:, -1]
                                X_test = df_test.iloc[:, :-1]
                                y_test = df_test.iloc[:, -1]

                                scaler = StandardScaler()
                                X_train_scaled = pd.DataFrame(
                                    scaler.fit_transform(X_train),
                                    columns = X_train.columns,
                                )
                                X_test_scaled = pd.DataFrame(
                                    scaler.transform(X_test),
                                    columns = X_test.columns,
                                )

                                y_scaler = StandardScaler()
                                y_train_scaled = pd.Series(
                                    y_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).ravel(),
                                    name = y_train.name,
                                )
                                y_test_scaled = pd.Series(
                                    y_scaler.transform(y_test.to_numpy().reshape(-1, 1)).ravel(),
                                    name = y_test.name,
                                )

                                scaled_train = pd.concat(
                                    [ X_train_scaled.reset_index(drop = True), y_train_scaled.reset_index(drop = True) ],
                                    axis = 1
                                )
                                scaled_test = pd.concat(
                                    [ X_test_scaled.reset_index(drop = True), y_test_scaled.reset_index(drop = True) ],
                                    axis = 1
                                )

                                model = model_class(
                                    scaled_train,
                                    scaled_test,
                                    config = variant_runtime_config,
                                    **variant_model_params,
                                )
                                model.fit()
                                y_pred = np.asarray(model.predict(), dtype = float).ravel()
                                y_true = y_test_scaled.to_numpy(dtype = float, copy = False)

                                if y_pred.shape[0] != y_true.shape[0]:
                                    raise ValueError(
                                        f"Prediction length mismatch: y_pred={y_pred.shape[0]}, y_true={y_true.shape[0]}"
                                    )
                                evaluator = Evaluator(y_pred, y_true)
                                split_results[idx] = {
                                    "MSE": evaluator.score_MSE(),
                                    "RMSE": evaluator.score_RMSE(),
                                    "MAE": evaluator.score_MAE(),
                                    "Adjusted R2 score": evaluator.score_r2(
                                        use_adjusted = True,
                                        num_feat = model.X_train.shape[1],
                                    ),
                                    "MAPE": evaluator.score_MAPE(),
                                    "sMAPE": evaluator.score_sMAPE(),
                                }
                            except Exception as e:
                                print(
                                    f"Unsuccessful evaluation due to '{e}'. Specifically, at model: {model_name}, "
                                    f"variant: {variant_name}, dataset: {dataset_name}, split type: {split_name}"
                                )
                                exit(0)

                        if split_results:
                            results[split_name] = split_results

                    output_file = model_output_dir / f"{dataset_name}.json"
                    with output_file.open("w", encoding = "utf-8") as f:
                        json.dump(results, f, indent = 2, default = _sanitize)

                    gc.collect()
