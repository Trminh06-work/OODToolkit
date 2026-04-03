from __future__ import annotations

import pandas as pd
import numpy as np

import json
import gc
from typing import List
from tqdm.notebook import tqdm

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
        rmse = mean_squared_error(self.y_true, self.y_pred)
        return rmse


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
        config = None
    ):
        self.models = models
        self.dataset_names = dataset_names
        self.source_dir = source_dir_location
        self.target_dir = result_dir_location
        self.config = ModelConfig if config is None else config


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

        for model_class in self.models:
            model_name = model_class.__name__
            print(f"Training model: {model_name}")

            model_output_dir = self.target_dir / model_name
            model_output_dir.mkdir(parents = True, exist_ok = True)

            dataset_dirs = sorted(path for path in self.source_dir.iterdir() if path.is_dir())
            if not dataset_dirs:
                raise ValueError(f"No split datasets found in {self.source_dir}")
            if self.dataset_names is not None:
                dataset_filter = set(self.dataset_names)
                dataset_dirs = [path for path in dataset_dirs if path.name in dataset_filter]

            for dataset_dir in dataset_dirs:
                dataset_name = dataset_dir.name
                results = {}
                print(f"  Dataset: {dataset_name}")

                for split_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
                    split_name = split_dir.name
                    train_files = sorted(split_dir.glob("train_*.parquet"))

                    if not train_files:
                        continue

                    split_results = {}
                    # print(f"    Split: {split_name}")

                    for train_file in train_files:
                        idx = train_file.stem.split("_")[1]
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

                            scaled_train = pd.concat(
                                [ X_train_scaled.reset_index(drop = True), y_train.reset_index(drop = True) ],
                                axis = 1
                            )
                            scaled_test = pd.concat(
                                [ X_test_scaled.reset_index(drop = True), y_test.reset_index(drop = True) ],
                                axis = 1
                            )

                            model = model_class(
                                scaled_train,
                                scaled_test,
                                config = self.config,
                            )
                            model.fit()
                            split_results[idx] = model.evaluate()
                        except Exception as e:
                            print(f"Unsuccessful evaluation due to '{e}'. Specifically, at model: {model_name}, dataset: {dataset_name}, split type: {split_name}")
                            exit(0)

                    if split_results:
                        results[split_name] = split_results

                output_file = model_output_dir / f"{dataset_name}.json"
                with output_file.open("w", encoding = "utf-8") as f:
                    json.dump(results, f, indent = 2, default = _sanitize)

                # Cleaning
                del df_train, df_test
                gc.collect()
