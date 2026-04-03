import pandas as pd
from .base_model import BaseModel, ModelConfig

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor



# ================================ Decision Tree-based Regressor ================================

class DTRegressor(BaseModel):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        max_depth = 12, min_samples_leaf = 20,
        min_samples_split = 40, max_features = "sqrt"
    ):
        """
        This class serves as a blueprint for the implementation of Decision Tree-based Regressor

        Parameters:
            df_train: pd.DataFrame, None as default. A training data saved as pandas DataFrame is required
            df_test: pd.DataFrame, None as default. A testing data saved as pandas DataFrame is required
            max_depth, min_samples_leaf, min_samples_split, max_features: referred to DecisionTreeRegressor from sklearn.tree
        """
        super().__init__(df_train, df_test, config)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.model = self.build_model()


    def build_model(self):
        return DecisionTreeRegressor(
            random_state = self.config.seed,
            max_depth = self.max_depth,
            min_samples_leaf = self.min_samples_leaf,
            min_samples_split = self.min_samples_split,
            max_features = self.max_features,
        )


    def fit(self):
        self.model.fit(self.X_train, self.y_train)


    def predict(self):
        return self.model.predict(self.X_test)


# ================================ Random Forest-based Regressor ================================

class RFRegressor(BaseModel):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        n_estimators = 400, max_samples = None, n_jobs = -1,
        max_depth = 12, min_samples_leaf = 20,
        min_samples_split = 40, max_features = "sqrt",
    ):
        """
        This class serves as a blueprint for the implementation of Random Forest-based Regressor

        Parameters:
            df_train: pd.DataFrame, None as default. A training data saved as pandas DataFrame is required
            df_test: pd.DataFrame, None as default. A testing data saved as pandas DataFrame is required
            n_estimators, max_samples, max_depth, min_samples_leaf, min_samples_split, max_features: referred to RandomForestRegressor from sklearn.ensemble
        """
        super().__init__(df_train, df_test, config)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.model = self.build_model()


    def build_model(self):
        return RandomForestRegressor(
            random_state = self.config.seed,
            n_estimators = self.n_estimators,
            max_depth = self.max_depth,
            min_samples_leaf = self.min_samples_leaf,
            min_samples_split = self.min_samples_split,
            max_features = self.max_features,
            max_samples = self.max_samples,
            bootstrap = True,
            n_jobs = self.n_jobs
        )


    def fit(self):
        self.model.fit(self.X_train, self.y_train)


    def predict(self):
        return self.model.predict(self.X_test)


# ================================ Gradient Boosted Tree-based Regressor ================================

class GBRegressor(BaseModel):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        n_estimators = 600,
        learning_rate = 0.05, max_depth = 3, subsample = 0.8,
        min_samples_leaf = 20, min_samples_split = 40,
        max_features = None,       # or "sqrt"/"log2"/float fraction
        tol = 1e-4
    ):
        """
        This class serves as a blueprint for the implementation of Gradient Boosted Tree-based Regressor

        Parameters:
            df_train: pd.DataFrame, None as default. A training data saved as pandas DataFrame is required
            df_test: pd.DataFrame, None as default. A testing data saved as pandas DataFrame is required
            n_estimators, min_samples_leaf, min_samples_split, max_features, learning_rate, subsample, tol: referred to GradientBoostingRegressor from sklearn.ensemble
        """
        super().__init__(df_train, df_test, config)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.tol = tol
        self.model = self.build_model()


    def build_model(self):
        return GradientBoostingRegressor(
            random_state = self.config.seed,
            n_estimators = self.n_estimators,
            learning_rate = self.learning_rate,
            max_depth = self.max_depth,
            subsample = self.subsample,
            min_samples_leaf = self.min_samples_leaf,
            min_samples_split = self.min_samples_split,
            max_features = self.max_features,
            tol = self.tol,
        )


    def fit(self):
        self.model.fit(self.X_train, self.y_train)


    def predict(self):
        return self.model.predict(self.X_test)


# ================================ Adaptive Boosted Tree (AdaBoost)-based Regressor ================================

class ABRegressor(BaseModel):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        max_depth = 12, min_samples_leaf = 20,
        min_samples_split = 40, max_features = "sqrt",
        n_estimators = 400, learning_rate = 0.05,
    ):
        """
        This class serves as a blueprint for the implementation of Adaptive Boosted Tree (AdaBoost)-based Regressor

        Parameters:
            df_train: pd.DataFrame, None as default. A training data saved as pandas DataFrame is required
            df_test: pd.DataFrame, None as default. A testing data saved as pandas DataFrame is required
            n_estimators, min_samples_leaf, min_samples_split, max_features, learning_rate, max_depth: referred to AdaBoostRegressor from sklearn.ensemble
        """
        super().__init__(df_train, df_test, config)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model = self.build_model()


    def build_model(self):
        base_estimator = DecisionTreeRegressor(
            random_state = self.config.seed,
            max_depth = self.max_depth,
            min_samples_leaf = self.min_samples_leaf,
            min_samples_split = self.min_samples_split,
            max_features = self.max_features,
        )

        return AdaBoostRegressor(
            random_state = self.config.seed,
            estimator = base_estimator,
            n_estimators = self.n_estimators,
            learning_rate = self.learning_rate,
        )


    def fit(self):
        self.model.fit(self.X_train, self.y_train)


    def predict(self):
        return self.model.predict(self.X_test)


# ================================ XGBoost-based Regressor ================================

class XGBRegressor(BaseModel):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        n_jobs = -1, n_estimators = 2000,
        learning_rate = 0.03, max_depth = 6,
        subsample = 0.8, colsample_bytree = 0.8,
        reg_lambda = 1.0, reg_alpha = 1.0, gamma = 1.0,
        max_bin = 64
    ):
        """
        This class serves as a blueprint for the implementation of XGBoost-based Regressor

        Parameters:
            df_train: pd.DataFrame, None as default. A training data saved as pandas DataFrame is required
            df_test: pd.DataFrame, None as default. A testing data saved as pandas DataFrame is required
            n_jobs, subsample, colsample_bytree, reg_lambda, reg_alpha, gamma, max_bin, n_estimators, learning_rate, max_depth: referred to xgb.XGBRegressor from xgboost
        """
        super().__init__(df_train, df_test, config)
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.max_bin = max_bin
        self.model = self.build_model()


    def build_model(self):
        return xgb.XGBRegressor(
            random_state = self.config.seed,
            n_jobs = self.n_jobs,
            max_depth = self.max_depth,
            n_estimators = self.n_estimators,
            learning_rate = self.learning_rate,
            subsample = self.subsample,
            colsample_bytree = self.colsample_bytree,
            reg_lambda = self.reg_lambda,
            reg_alpha = self.reg_alpha,
            gamma = self.gamma,
            max_bin = self.max_bin,
        )


    def fit(self):
        self.model.fit(self.X_train, self.y_train)


    def predict(self):
        return self.model.predict(self.X_test)


# ================================ LightGBM-based Regressor ================================

class LightGBMRegressor(BaseModel):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        n_estimators = 2000,        # large; rely on early stopping
        learning_rate = 0.03,
        num_leaves = 63,              # for <20 features
        max_depth = -1,               # -1 = no limit
        min_child_samples = 50,       # a.k.a. min_data_in_leaf
        subsample = 0.8,              # bagging_fraction
        colsample_bytree = 0.8,       # feature_fraction
        reg_lambda = 1.0,
        reg_alpha = 0.0,
        # Speed / stability
        n_jobs = -1,
    ):
        """
        This class serves as a blueprint for the implementation of XGBoost-based Regressor

        Parameters:
            df_train: pd.DataFrame, None as default. A training data saved as pandas DataFrame is required
            df_test: pd.DataFrame, None as default. A testing data saved as pandas DataFrame is required
            n_jobs, subsample, colsample_bytree, reg_lambda, reg_alpha, n_estimators, learning_rate, max_depth, num_leaves, min_child_samples: referred to LGBMRegressor from lightgbm
        """
        super().__init__(df_train, df_test, config)
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.model = self.build_model()


    def build_model(self):
        return LGBMRegressor(
            verbosity = -1,
            random_state = self.config.seed,
            n_jobs = self.n_jobs,
            max_depth = self.max_depth,
            n_estimators = self.n_estimators,
            learning_rate = self.learning_rate,
            subsample = self.subsample,
            colsample_bytree = self.colsample_bytree,
            reg_lambda = self.reg_lambda,
            reg_alpha = self.reg_alpha,
            num_leaves = self.num_leaves,
            min_child_samples = self.min_child_samples,
        )


    def fit(self):
        self.model.fit(self.X_train, self.y_train)


    def predict(self):
        return self.model.predict(self.X_test)