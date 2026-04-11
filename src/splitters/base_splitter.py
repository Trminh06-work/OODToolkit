from typing import List
from abc import ABC, abstractmethod
import pandas as pd

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


    def _preprocess(self):
        if self.df.shape[0] > 1000000 and not self.keep_size:
            self.df = self.df.sample(n = 800000, random_state = 42).reset_index(drop=True)
            print("Remove some samples due to extensive size")
            print(f"New Data: {self.df.shape[0]} samples, {self.df.shape[1] - 1} features")


    @abstractmethod
    def split(self, file_name: str = None, df: pd.DataFrame = None, test_size: float = None, include_reverse: bool = False):
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
