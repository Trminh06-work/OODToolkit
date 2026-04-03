from .base_model import BaseModel, ModelConfig, pick_device
from .statistical_models import HuberLinearRegressor, HuberPolynomialRegressor, SVMRegressor, KNNRegressor
from .tree_models import DTRegressor, RFRegressor, GBRegressor, ABRegressor, XGBRegressor, LightGBMRegressor
from .resnet import ResnetRegressor