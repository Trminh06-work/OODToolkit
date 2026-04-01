from .base_splitter import BaseSplitter

import os
import numpy as np
import pandas as pd

class MarginalDistributionSplit(BaseSplitter):
    """
    Note that this splitter does not depend on the SEEDs
    """
    def __init__(self, seeds = None, keep_size = False):
        super().__init__(seeds, keep_size)


    def _distribution_based_selection(self, feat):
        """
            This function returns a permutatation of which rows are selected

            Parameters
                feat      :    the feature column being considered

            Return:
                col_mask  :    a permuation of selected samples' indices
        """
        lo = 0
        hi = 1
        epsilon = 0.001

        while lo < hi:
            mid = lo + (hi - lo) / 2

            digitized_col = np.digitize(self.df.loc[:, feat], np.quantile(self.df.loc[:, feat], [mid, 1 - mid]))

            # col_mask keeps the bits 1 on the row that a specific column is numbered 1, otherwise 0.
            col_mask = (digitized_col == 1)

            if col_mask.sum() / self.df.shape[0] < self.test_size:
                lo = mid + epsilon
            else:
                hi = mid - epsilon

        return col_mask


    def _covariate_shift(self):
        """
            Repeatedly loop through all features, based on predefined TEST_SIZE/TRAIN_SIZE and their distributions to split the dataset.
        """

        # Create directory if not exist
        output_dir = f"../data/splitted/{self.file_name}/Covariate_Shift"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        idx = 0

        for feat in self.df.columns[:-1]:
            mask = self._distribution_based_selection(feat)

            if mask.sum() / self.df.shape[0] < self.test_size:
                print(f"Skip {feat} feature due to insufficient test samples")
                continue

            X_train = self.X[~mask]
            y_train = self.y[~mask]
            X_test = self.X[mask]
            y_test = self.y[mask]

            df_train = pd.concat([X_train, y_train], axis = 1)
            df_test = pd.concat([X_test, y_test], axis = 1)

            # Save files using the idx
            path = os.path.join(output_dir, f"train_{idx}.parquet")
            df_train.to_parquet(path, index = False)
            path = os.path.join(output_dir, f"test_{idx}.parquet")
            df_test.to_parquet(path, index = False)

            idx += 1


    def split(self, file_name = None, df = None, test_size = None):
        super().split(file_name, df, test_size)

        self._covariate_shift()

