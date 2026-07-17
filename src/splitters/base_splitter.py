import os
from typing import List
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import random
from pathlib import Path

class BaseSplitter:
    def __init__(self, seeds: List[int] = None, keep_size: bool = False):
        """
            seeds: the random seeds for consistency across experiment and reproducibility

            keep_size: (default: False) -> set to True to keep the big-sized data, >1M samples
        """
        if seeds is None:
            raise ValueError("No seeds are given")
        self.SEEDS = seeds
        self.keep_size = keep_size
        self.add_noise = False


    def _preprocess(self):
        if self.df.shape[0] > 1000000 and not self.keep_size:
            self.df = self.df.sample(n = 800000, random_state = 42).reset_index(drop=True)
            print("Remove some samples due to extensive size")
            print(f"New Data: {self.df.shape[0]} samples, {self.df.shape[1] - 1} features")


    def save_data(self, df_train: pd.DataFrame, df_test: pd.DataFrame, output_dir, idx):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents = True, exist_ok = True)

        # Add noise to train
        if self.add_noise:
            df_train = df_train.copy(deep = True)
            if idx < len(self.SEEDS):
                noise_seed = self.SEEDS[idx]
            else:
                # Covariate Shift splits based on quantile, no seed is provided
                if output_dir.name != "Covariate_Shift":
                    # print(output_dir)
                    raise ValueError("idx is beyond the existing SEEDS")
                noise_seed = 42 # Default for Covariate Shift only

            random.seed(noise_seed)
            np.random.seed(noise_seed)
            sigma = 1/16 # sigma or standard deviation
            df_train.iloc[:, -1] += np.random.normal(
                loc = 0,
                scale = sigma,
                size = df_train.shape[0]
            )
        path = os.path.join(output_dir, f"train_{idx}.parquet")
        df_train.to_parquet(path, index = False)
        path = os.path.join(output_dir, f"test_{idx}.parquet")
        df_test.to_parquet(path, index = False)


    @abstractmethod
    def split(self, file_name: str = None, df: pd.DataFrame = None, test_size: float = None,
              include_reverse: bool = False, add_noise_to_train: bool = False
    ):
        if file_name is None:
            raise ValueError("file_name is not given")
        if df is None:
            raise ValueError("No data is given")
        if test_size is None or test_size < 0 or test_size > 1:
            raise ValueError("test_size is not specified or incorrectly given")
        self.file_name = file_name
        self.df = df
        self.test_size = test_size

        self._preprocess()
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]
        self.add_noise = add_noise_to_train
