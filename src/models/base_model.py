from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

@dataclass
class ModelConfig:
    use_optim: bool = False
    metric: str = "rmse"
    n_splits: int = 5
    seed: int = 42

# ================================== DL Auxiliary Classes and Functions ==================================

def pick_device(prefer_mps: bool = True, prefer_cuda: bool = True) -> str:
    """Pick the best available accelerator with a slight preference ordering."""
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    if prefer_mps and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalization(norm: str, dim: int):
    norm = norm.lower()
    match norm:
        case "batchnorm1d":
            return nn.BatchNorm1d(dim)
        case "layernorm":
            return nn.LayerNorm(dim)
        case "" | "none":
            return nn.Identity()
        case _:
            raise ValueError(f"Unknown norm: {norm}")


def activation_fn(act_name: str):
    act_name = act_name.lower()
    match act_name:
        case "relu":
            return nn.ReLU()
        case "gelu":
            return nn.GELU()
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()
        case "silu" | "swish":
            return nn.SiLU()
        case _:
            raise ValueError(f"Unknown activation function: {act_name}")


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



class CreateDataLoader:
    def __init__(
        self,
        X_train, X_test,
        y_train, y_test,
        batch_size: int = 512,
        val_size: float = 0.1,
        is_shuffle = True,
        seed: int = 42,
    ):
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size = val_size,
            shuffle = is_shuffle,
            random_state = seed,
        )

        # Save
        self.X_train = self._to_tensor(X_train)
        self.X_val = self._to_tensor(X_val)

        # Do not convert X_test and y_test, keep it in pd.DataFrame to plot it later
        self.X_test_tensor = self._to_tensor(X_test)

        self.y_train = self._to_tensor(y_train)
        self.y_val   = self._to_tensor(y_val)

        self.batch_size = batch_size
        self.is_shuffle = is_shuffle


    def _to_tensor(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype = np.float32, copy=False)
        else:
            X = np.asarray(X, dtype = np.float32)

        if X.ndim == 1:
            X = X.reshape(-1)

        return torch.from_numpy(X)


    def create(self):
        train_dataset = CustomDataset(self.X_train, self.y_train)
        val_dataset = CustomDataset(self.X_val, self.y_val)

        # Avoid a last mini-batch of size 1 (BatchNorm crashes in train mode)
        drop_last = (len(train_dataset) > self.batch_size) and (len(train_dataset) % self.batch_size == 1)

        train_loader = DataLoader(
            train_dataset,
            batch_size = self.batch_size,
            shuffle = self.is_shuffle,
            drop_last = drop_last,
            num_workers = 4,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size = self.batch_size,
            shuffle = not self.is_shuffle,
            num_workers = 4,
        )

        return train_loader, val_loader

# ================================== BaseModel Abstract Class ==================================

class BaseModel(ABC):
    def __init__(
            self,
            df_train: pd.DataFrame = None, df_test: pd.DataFrame = None,
            config: ModelConfig = None,
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

        self.config = config if config is not None else ModelConfig()


    @abstractmethod
    def fit(self):
        raise NotImplementedError


    @abstractmethod
    def predict(self):
        raise NotImplementedError


    def evaluate(self) -> Dict[str, float]:
        # avoid circular import warnings
        from benchmark import Evaluator

        y_pred = self.predict()
        evaluator = Evaluator(y_pred, self.y_test)
        return {
            "MSE": evaluator.score_MSE(),
            "RMSE": evaluator.score_RMSE(),
            "MAE": evaluator.score_MAE(),
            "Adjusted R2 score": evaluator.score_r2(use_adjusted = True, num_feat = self.X_train.shape[1]),
            "MAPE": evaluator.score_MAPE(),
            "sMAPE": evaluator.score_sMAPE()
        }


# ================================== BaseDLModel Abstract Class ==================================

class BaseDLModel(BaseModel):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        epochs: int = 100,
        batch_size: int = 1024,
        seed: int = 42,
        optimizer = None,
        loss_criterion = None,
        plot_train_progress: bool = False
    ):
        super().__init__(df_train, df_test, config)
        self.device = pick_device()
        self.epochs = epochs
        self.batch_size = batch_size
        self.plot_train_progress = plot_train_progress

        self.num_feat = self.X_train.shape[1]
        self.criterion = nn.MSELoss() if loss_criterion is None else loss_criterion
        self.model = self.model.to(self.device)
        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr = self.lr,
                weight_decay = self.weight_decay
            )
        else:
            self.optimizer = optimizer

        if df_test is not None:
            DataloaderCreator = CreateDataLoader(
                self.X_train, self.X_test, self.y_train, self.y_test,
                batch_size = batch_size, seed = seed
            )
            self.train_loader, self.val_loader = DataloaderCreator.create()
            self.X_test_tensor = DataloaderCreator.X_test_tensor.to(self.device)


    def _to_numpy(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)


    def _to_tensor(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float32, copy=False)
        else:
            X = np.asarray(X, dtype=np.float32)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return torch.from_numpy(X).to(self.device)


    def _score_r2(self, y_true, y_pred, use_adjusted = True, num_feat = None):
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        r2 = r2_score(y_true, y_pred)
        if use_adjusted:
            if num_feat is None:
                print("missing number of features")
                return None
            n = len(y_true)
            r2 = 1 - ((1 - r2) * ((n - 1) / (n - num_feat - 1)))
        return r2


    def get_params(self):
        return None


    def fit(self):
        train_losses = []
        val_losses = []
        train_adjusted_r2 = []
        val_adjusted_r2 = []

        X = self._to_tensor(self.X_train)
        y = self._to_tensor(self.y_train)
        train_dataset = CustomDataset(X, y)

        # Avoid a last mini-batch of size 1 (BatchNorm crashes in train mode)
        drop_last = (len(train_dataset) > self.batch_size) and (len(train_dataset) % self.batch_size == 1)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            drop_last = drop_last,
            num_workers = 0,
        )

        for epoch in range(self.epochs):
            if self.plot_train_progress:
                train_loss = 0.0
                train_target = []
                train_predict = []

            # Training
            self.model.train()
            for X_samples, y_samples in self.train_loader:
                X_samples, y_samples = X_samples.to(self.device), y_samples.to(self.device)
                self.optimizer.zero_grad()
                y_preds = self.model(X_samples).view(-1)
                loss = self.criterion(y_preds, y_samples)
                loss.backward()
                self.optimizer.step()
                if self.plot_train_progress:
                    train_target.append(y_samples.detach())
                    train_predict.append(y_preds.detach())
                    train_loss += loss.item()


            # Skip if this is unecessary
            if self.plot_train_progress:
                train_loss /= len(self.train_loader)
                train_losses.append(train_loss)

                train_target_t = torch.cat(train_target, dim=0)
                train_predict_t = torch.cat(train_predict, dim=0)
                train_adjusted_r2.append(self._score_r2(
                    y_true=train_target_t, y_pred=train_predict_t,
                    use_adjusted=True, num_feat=self.num_feat
                ))
                # Evaluation
                val_loss = 0.0
                val_target = []
                val_predict = []

                self.model.eval()
                with torch.no_grad():
                    for X_samples, y_samples in self.val_loader:
                        X_samples, y_samples = X_samples.to(self.device), y_samples.to(self.device)
                        y_preds = self.model(X_samples)
                        loss = self.criterion(y_preds, y_samples)

                        val_loss += loss.item()
                        val_target.append(y_samples.detach())
                        val_predict.append(y_preds.detach())

                val_loss /= len(self.val_loader)
                val_losses.append(val_loss)

                val_target_t = torch.cat(val_target, dim=0)
                val_predict_t = torch.cat(val_predict, dim=0)
                val_adjusted_r2.append(self._score_r2(
                    y_true=val_target_t, y_pred=val_predict_t,
                    use_adjusted=True, num_feat=self.num_feat
                ))
                if (epoch + 1) % 10 == 0:
                    print(f'\nEPOCH {epoch + 1}:\tTraining loss: {train_loss:.3f}\tValidation loss: {val_loss:.3f}')

        if self.plot_train_progress:
            self._plot_train_progress(
                train_losses, val_losses,
                train_adjusted_r2, val_adjusted_r2
            )

        return self


    def predict(self) -> pd.Series:
        self.model.eval()
        index = None
        with torch.no_grad():
            X_test = self._to_tensor(self.X_test)

            # Avoid a last mini-batch of size 1 (BatchNorm crashes in train mode)
            loader = DataLoader(TensorDataset(X_test), batch_size = self.batch_size, shuffle = False)

            preds = []
            with torch.no_grad():
                for (Xb,) in loader:
                    Xb = Xb.to(self.device)
                    yb = self.model(Xb).view(-1).detach().cpu()
                    preds.append(yb)

            y_pred = torch.cat(preds).numpy()

            # if index is not None:
            return pd.Series(y_pred, name = "y_pred")
            # return y_pred


    def _plot_train_progress(
        self,
        train_losses, val_losses,
        train_r2, val_r2
    ):
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        ax[0, 0].plot(train_losses, color='green')
        ax[0, 0].set(xlabel='Epoch', ylabel='Loss')
        ax[0, 0].set_title('Training Loss')
        ax[0, 0].grid(True, linestyle = "--", alpha = 0.5)

        ax[0, 1].plot(val_losses, color='orange')
        ax[0, 1].set(xlabel='Epoch', ylabel='Loss')
        ax[0, 1].set_title('Validation Loss')
        ax[0, 1].grid(True, linestyle = "--", alpha = 0.5)

        ax[1, 0].plot(train_r2, color='green')
        ax[1, 0].set(xlabel='Epoch', ylabel='R2')
        ax[1, 0].set_title('Training R2')
        ax[1, 0].grid(True, linestyle = "--", alpha = 0.5)

        ax[1, 1].plot(val_r2, color='orange')
        ax[1, 1].set(xlabel='Epoch', ylabel='R2')
        ax[1, 1].set_title('Validation R2')
        ax[1, 1].grid(True, linestyle = "--", alpha = 0.5)

        plt.show()