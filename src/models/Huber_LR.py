from .base_model import BaseModel, ModelConfig

import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
class HuberLinearRegressor(BaseModel):
    def __init__(
    self,
    df_train: pd.DataFrame, df_test: pd.DataFrame,
    config: ModelConfig,
    epsilon = 1.35, alpha = 1e-4,
    max_iter = 5000, tol = 1e-4,
    l1_ratio = 0.2
    ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.l1_ratio = l1_ratio
        self.config = config
        self.model = self._build_model()
        super().__init__(df_train, df_test) # construct (X_train, y_train) and (X_test, y_test)


    def _build_model(self):
        # Scale both X and y so the SGD step sizes stay stable across datasets
        return SGDRegressor(
            loss = "huber",
            penalty = "elasticnet",
            l1_ratio = self.l1_ratio,
            epsilon = self.epsilon,
            alpha = self.alpha,
            max_iter = self.max_iter,
            tol = self.tol,
            random_state = self.config.seed,
        )