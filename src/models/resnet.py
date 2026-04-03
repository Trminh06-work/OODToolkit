from .base_model import BaseDLModel, ModelConfig, normalization, activation_fn

import pandas as pd
import torch.nn as nn



class ResBlock(nn.Module):
    def __init__(
        self,
        d: int,
        d_hidden: int,
        act_fn: str = "relu",
        norm: str = "batchnorm1d",
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.norm = normalization(norm, d)
        self.fc1 = nn.Linear(d, d_hidden)
        self.activation = activation_fn(act_fn)
        self.drop = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(d_hidden, d)


    def forward(self, x):
        identity = x     # For Skip Connection
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        x = x + identity  # Skip Connection

        return x



class ResNet(nn.Module):
    def __init__(
        self,
        d_in: int,
        d: int = 256,
        n_res_blocks: int = 8,
        d_out: int = 1,
        d_hidden_factor: float = 0.2,
        dropout_rate: float = 0.2,
        act_fn: str = "relu",
        norm: str = "batchnorm1d",

        # Training hyperparams
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 1024
    ):
        super().__init__()
        d_hidden = int(d * d_hidden_factor)
        self.fc0 = nn.Linear(d_in, d)

        self.resnet_layers = nn.ModuleList(
            [
                ResBlock(
                    d = d,
                    d_hidden = d_hidden,
                    act_fn = act_fn,
                    norm = norm,
                    dropout_rate = dropout_rate
                )
                for _ in range(n_res_blocks)
            ]
        )

        self.prediction = nn.Sequential(
            normalization(norm, d),
            activation_fn(act_fn),
            nn.Linear(d, d_out)
        )

        self.param_dict = {
            "d_in": d_in,
            "d": d,
            "n_res_blocks": n_res_blocks,
            "d_out": d_out,
            "d_hidden_factor": d_hidden_factor,
            "dropout_rate": dropout_rate,
            "act_fn": act_fn,
            "norm": norm,

            # Training hyperparams
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size
        }


    def forward(self, x):
        x = self.fc0(x)
        for layer in self.resnet_layers:
            x = layer(x)
        x = self.prediction(x)
        return x.squeeze(-1)


    def get_params(self):
        return self.param_dict



class ResnetRegressor(BaseDLModel):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig, optimizer = None, loss_criterion = None,
        plot_train_progress: bool = False,

        # ResNet
        random_state: int = 42,
        d_in: int = 1,
        d: int = 512,
        n_res_blocks: int = 4,
        d_out: int = 1,
        d_hidden_factor: float = 0.2,
        dropout_rate: float = 0.2,
        act_fn: str = "relu",
        norm: str = "batchnorm1d",

        # Training hyperparams
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 1024,
    ):
        self.param_dict = {
            "d_in": df_train.shape[1] - 1,
            "d": d,
            "n_res_blocks": n_res_blocks,
            "d_out": d_out,
            "d_hidden_factor": d_hidden_factor,
            "dropout_rate": dropout_rate,
            "act_fn": act_fn,
            "norm": norm,

            # Training hyperparams
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size
        }

        self.lr = lr
        self.weight_decay = weight_decay
        self.model = self.build_model()
        super().__init__(df_train, df_test, config, epochs, batch_size, random_state, optimizer, loss_criterion, plot_train_progress)


    def build_model(self):
        return ResNet(**self.param_dict)