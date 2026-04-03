from .base_model import BaseModel, ModelConfig

import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor


# ================================ Linear Regressor using Huber Loss ================================

class HuberLinearRegressor(BaseModel):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig = None,
        epsilon = 1.35, alpha = 1e-4,
        max_iter = 5000, tol = 1e-4,
        l1_ratio = 0.2
    ):
        """
        This class serves as a blueprint for the implementation of Linear Regressor using Huber Loss

        Parameters:
            df_train: pd.DataFrame, None as default. A training data saved as pandas DataFrame is required
            df_test: pd.DataFrame, None as default. A testing data saved as pandas DataFrame is required
            epsilon, alpha, max_iter, tol, l1_ratio: referred to SGDRegressor implemented in sklearn.linear_model
        """
        super().__init__(df_train, df_test, config = config) # construct (X_train, y_train) and (X_test, y_test)
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.l1_ratio = l1_ratio
        self.model = self._build_model()


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


    def fit(self):
        self.model.fit(self.X_train, self.y_train)


    def predict(self):
        return self.model.predict(self.X_test)


# ================================ Polynomial Regressor using Huber Loss ================================
class HuberPolynomialRegressor(BaseModel):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig = None,
        degree: int = 2,
        epsilon = 1.35, alpha = 5e-4,
        max_iter = 5000, tol = 1e-4,
        l1_ratio = 0.25
    ):
        """
        This class serves as a blueprint for the implementation of Polynomial Regressor using Huber Loss

        Parameters:
            df_train: pd.DataFrame, None as default. A training data saved as pandas DataFrame is required
            df_test: pd.DataFrame, None as default. A testing data saved as pandas DataFrame is required
            degree: int, 2 as default. Specify the degree of polynomial expression
            epsilon, alpha, max_iter, tol, l1_ratio: referred to SGDRegressor implemented in sklearn.linear_model
        """
        super().__init__(df_train, df_test, config = config)
        self.degree = degree
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.l1_ratio = l1_ratio
        self.model = self.build_model()


    def build_model(self):
        # Scale both X and y to stabilize SGD across datasets
        return make_pipeline(
            PolynomialFeatures(
                degree = self.degree,
                include_bias = False,
                interaction_only = True, # reduces the impact of a single feature
            ),
            SGDRegressor(
                loss = "huber",
                penalty = "elasticnet",
                epsilon = self.epsilon,
                alpha = self.alpha,
                max_iter = self.max_iter,
                tol = self.tol,
                l1_ratio = self.l1_ratio,
                random_state = self.config.seed,
            )
        )


    def fit(self):
        self.model.fit(self.X_train, self.y_train)


    def predict(self):
        return self.model.predict(self.X_test)


# ================================ KNN Regressor ================================

class KNNRegressor(BaseModel):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig = None,
        weights: str = "distance",     # uniform or distance
        n_neighbors: int = 5
    ):
        """
        This class serves as a blueprint for the implementation of KNN Regressor

        Parameters:
            df_train: pd.DataFrame, None as default. A training data saved as pandas DataFrame is required
            df_test: pd.DataFrame, None as default. A testing data saved as pandas DataFrame is required
            weights, n_neighbors: referred to KNeighborsRegressor implemented in sklearn.neighbors
        """
        super().__init__(df_train, df_test, config = config)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.model = self.build_model()


    def build_model(self):
        return KNeighborsRegressor(
            n_neighbors = self.n_neighbors,
            weights = self.weights,
            algorithm = "auto"
        )


    def fit(self):
        self.model.fit(self.X_train, self.y_train)


    def predict(self):
        return self.model.predict(self.X_test)


# ================================ Linear Regressor using epsilon-insensitive Loss ================================


class SVMRegressor(BaseModel):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig = None,
        epsilon = 0.2, alpha = 1e-4,
        max_iter = 5000, tol = 1e-4,
    ):
        super().__init__(df_train, df_test, config = config)
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.model = self.build_model()


    def build_model(self):
        # Scale both X and y to keep epsilon-insensitive SGD stable
        return SGDRegressor(
            loss = "epsilon_insensitive",
            epsilon = self.epsilon,
            alpha = self.alpha,
            max_iter = self.max_iter,
            tol = self.tol,
            penalty = "l2",
            learning_rate = "optimal",
            random_state = self.config.seed,
        )


    def fit(self):
        self.model.fit(self.X_train, self.y_train)


    def predict(self):
        return self.model.predict(self.X_test)
