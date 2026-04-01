from __future__ import annotations

import os
import pandas as pd
import numpy as np

import json
from tqdm.notebook import tqdm
from pathlib import Path
import gc
import copy
from collections import defaultdict
from typing import Dict

from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

from sklearn.preprocessing import StandardScaler

from ..models import BaseModel, ModelConfig

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



class DataSaver:
    def __init__(self, model_name):
        self.model_name = model_name
        self.output_dir = "Results/"
        os.makedirs(self.output_dir, exist_ok = True)


    def _to_python(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if hasattr(obj, "item"):   # torch / numpy scalar
            return obj.item()
        raise TypeError


    def save_result(self, out_file, results):
        # Save path:  Results/{model_name}/{file_name}.json
        # where in each .json file, it encompasses of results for all types of split
        try:
            with open(out_file, "w") as f:
                json.dump(results, f, indent = 2, default = self._to_python)
            # tqdm.write(f"Successfully saved → {out_file}")
        except:
            tqdm.write(f"Error: Cannot save file")


    def read_json(self, file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
        return defaultdict(dict, data)


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
    def __init__(self, model_name: str = None, MODEL_REGISTRY: Dict[str, BaseModel] = None):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")
        self.model_name = model_name
        self.model_class = MODEL_REGISTRY[model_name]
        self.config = ModelConfig


    def _process(self, df_train, df_test):
        config = copy.deepcopy(self.config)

        if self.model_class is ResnetRegressor:
            regressor = ResnetRegressor(df_train, df_test, config, d_in = df_train.shape[1] - 1)
        elif self.model_class is FTTransformerRegressor:
            regressor = FTTransformerRegressor(df_train, df_test, config, d_in = df_train.shape[1] - 1, n_blocks = 3)
        else:
            regressor = self.model_class(df_train, df_test, config)

        regressor.fit()
        result = regressor.evaluate()

        return result


    def evaluate(self, ds_lst = DATASET_LIST):
        path = f"../data/splitted"
        data_saver = DataSaver(self.model_name)

        for file_name in tqdm(ds_lst, desc = "Dataset processing"):
            results = defaultdict(dict)

            save_dir = os.path.join("Results_add/", self.model_name)
            os.makedirs(save_dir, exist_ok=True)
            out_file = os.path.join(save_dir, f"{file_name}.json")

            if os.path.exists(out_file):
                results = data_saver.read_json(out_file)
                if len(results) == len(ds_lst):
                    # tqdm.write("Skip due to file exists")
                    continue

            for split_type in tqdm(SPLIT_TYPES, desc = f"Processing {file_name} splits", leave = False):
                folder = Path(os.path.join(path, file_name, split_type))
                train_files = sorted(folder.glob("train_*.parquet"))

                if split_type in results:
                    # tqdm.write("Skip due to split type exists")
                    continue

                for train_file in tqdm(train_files, desc = f"{file_name}/{split_type}", leave = False):
                    idx = train_file.stem.split("_")[1]
                    test_file = folder / f"test_{idx}.parquet"

                    if not test_file.exists():
                        tqdm.write(f"Warning: test file missing for idx={idx}")
                        continue

                    try:
                        scaler = StandardScaler()
                        df_train = pd.read_parquet(train_file)
                        df_test = pd.read_parquet(test_file)

                        df_train = pd.DataFrame(
                            scaler.fit_transform(df_train),
                            columns = df_train.columns
                        )
                        df_test = pd.DataFrame(
                            scaler.transform(df_test),
                            columns = df_test.columns
                        )
                    except Exception as e:
                        tqdm.write(f"Read failed for idx = {idx}: {e}")
                        continue

                    results[split_type][idx] = self._process(df_train, df_test)

                    del df_train, df_test
                    gc.collect()

                data_saver.save_result(out_file, results)