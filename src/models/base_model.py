from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import pandas as pd
from ..benchmark import Evaluator

from typing import Dict
@dataclass
class ModelConfig:
    use_optim: bool = False
    metric: str = "rmse"
    n_splits: int = 5
    seed: int = 42


def pick_device(prefer_mps: bool = True, prefer_cuda: bool = True) -> str:
    """Pick the best available accelerator with a slight preference ordering."""
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    if prefer_mps and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BaseModel(ABC):
    def __init__(
            self,
            df_train: pd.DataFrame, df_test: pd.DataFrame
    ):
        if df_train is None or df_test is None:
            raise ValueError("train or test files are None")

        self.df_train = df_train
        self.df_test = df_test

        # Split X/y (assumes last column is target)
        self.X_train = df_train.iloc[:, :-1]
        self.y_train = df_train.iloc[:, -1]
        self.X_test = df_test.iloc[:, :-1]
        self.y_test = df_test.iloc[:, -1]


    @abstractmethod
    def fit():
        raise NotImplementedError


    @abstractmethod
    def predict():
        raise NotImplementedError


    def evaluate(self) -> Dict[str, float]:
        y_pred = self.predict()
        evaluator = Evaluator(y_pred, self.y_test)
        return {
            "MSE": evaluator.score_MSE(),
            "RMSE": evaluator.score_RMSE(),
            "MAE": evaluator.score_MAE(),
            "Adjusted R2 score": evaluator.score_r2(
                use_adjusted=True, num_feat=self.X_train.shape[1]
            ),
            "MAPE": evaluator.score_MAPE(),
            "sMAPE": evaluator.score_sMAPE()
        }