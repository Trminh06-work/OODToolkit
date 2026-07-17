from .base_model import BaseModel, pick_device
from sklearn.neural_network import MLPRegressor
from tabicl import TabICLRegressor
from pytabkit import RealMLP_TD_Regressor

# Sklearn MLP: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
# All default values, except max_iter for higher precision if possible
# Optimizer used: Adam
class BaselineMLPRegressor(BaseModel):
    def __init__(
        self, df_train = None, df_test = None, config = None,
        random_state: int = 42, max_iter: int = 2000, batch_size: int = 1024
    ):
        super().__init__(df_train, df_test, config)
        self.model = MLPRegressor(random_state = random_state, max_iter = max_iter, batch_size = batch_size)


    def fit(self):
        self.model.fit(self.X_train, self.y_train)


    def predict(self):
        return self.model.predict(self.X_test)


# TabICL v1: https://arxiv.org/abs/2502.05564
# TabICL v2: https://arxiv.org/abs/2602.11139 (this implementation)
# Architecture: col-wise Transformer (embed features) -> row-wise Transformer (agg features) -> dataset-wise Transformer (In-context learning)
class TabiclRegressor(BaseModel):
    def __init__(
        self, df_train = None, df_test = None, config = None,
        random_state: int = 42, batch_size: int = 1024, kv_cache = False
    ):
        super().__init__(df_train, df_test, config)
        # kv_cache for faster inference -> more use of memory
        self.model = TabICLRegressor(
            random_state = random_state, batch_size = batch_size, device = pick_device(), kv_cache = kv_cache)

    def fit(self):
        self.model.fit(self.X_train, self.y_train)


    def predict(self):
        return self.model.predict(self.X_test)


# RealMLP Regressor (NeurIPS 2024): https://arxiv.org/abs/2407.04491
class RealMLPRegressor(BaseModel):
    def __init__(
        self, df_train = None, df_test = None, config = None,
        random_state: int = 42, batch_size: int = 1024
    ):
        super().__init__(df_train, df_test, config)
        self.model = RealMLP_TD_Regressor(device = pick_device(), random_state = random_state, batch_size = batch_size)


    def fit(self):
        self.model.fit(self.X_train, self.y_train)


    def predict(self):
        return self.model.predict(self.X_test)