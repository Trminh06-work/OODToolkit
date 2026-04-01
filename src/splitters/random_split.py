import pandas as pd
import os
from sklearn.model_selection import train_test_split
from .base_splitter import BaseSplitter


class RandomSplit(BaseSplitter):
    def __init__(self, seeds = None, keep_size = False):
        super().__init__(seeds, keep_size)


    def _random_split(self, SEEDS):
        """
            keep_size: (default: False) -> set to True to keep the big-sized data, >1M samples
        """
        # Create directory if not exist
        output_dir = f"../data/splitted/{self.file_name}/Random_Split"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self._preprocess()

        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]

        for idx, seed in enumerate(SEEDS):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size = self.test_size,
                random_state = seed,
                shuffle = True
            )

            df_train = pd.concat([X_train, y_train], axis = 1)
            df_test = pd.concat([X_test, y_test], axis = 1)

            # Save files using the idx
            path = os.path.join(output_dir, f"train_{idx}.parquet")
            df_train.to_parquet(path, index = False)
            path = os.path.join(output_dir, f"test_{idx}.parquet")
            df_test.to_parquet(path, index = False)


    def split(self, file_name = None, df = None, test_size = None):
        super().split(file_name, df, test_size)

        self._random_split()