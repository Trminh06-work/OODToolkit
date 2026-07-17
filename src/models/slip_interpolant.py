from .base_model import BaseModel, ModelConfig
import numpy as np
from .liblipt import LibLipT
import torch
import pandas as pd

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SLipInterpolant(BaseModel):
    def __init__(self,
        df_train: pd.DataFrame = None, df_test: pd.DataFrame = None, config: ModelConfig = None,
        type: str = "baseline", mon: np.array = None, num_k: int = 1,
        a = None, b = None, w = None
    ):
        super().__init__(df_train, df_test, config)

        if type not in ["baseline", "baseline_mon", "baseline_mix", "local", "local_mon", "local_mix", "smooth"]:
            raise ValueError("type must be `baseline`, `baseline_mon`, `baseline_mix`, `local`, `local_mon`, `local_mix` or `smooth`")

        if type not in ["baseline", "local"] and mon is None:
            raise ValueError(f"mon must be clearly specified since type = {type}")
        else:
            self.mon = mon

        # Construct the Slip interpolant using BaseModel's X/y split
        self.dim = int(self.X_train.shape[1])
        self.npts = int(self.X_train.shape[0])
        self.type = type
        self.num_k = num_k


        if type in ["baseline_mon_bound", "local_mon_bound"]:
            if a is None or b is None or w is None:
                raise ValueError("a, b, and w must be specified")
        else:
            a = np.zeros(self.dim, dtype = float)
            b = np.zeros(self.dim)
            w = np.zeros(self.dim)

        self.a = a
        self.b = b
        self.w = w

        # Convert to NumPy arrays -> flatten for compatible with the liblip codebase
        self.X_train_np = np.ascontiguousarray(
            self.X_train.to_numpy(dtype = np.float32, copy = False)
        )
        self.y_train_np = np.ascontiguousarray(
            self.y_train.to_numpy(dtype = np.float32, copy = False)
        )
        self.X_test_np = np.ascontiguousarray(
            self.X_test.to_numpy(dtype = np.float32, copy = False)
        )

        self.sli = LibLipT(capacity = self.npts + 7, dim = self.dim, knn = self.num_k, device = get_device())
        self.sli.add(self.X_train_np, self.y_train_np)


    def fit(self):
        if self.type in ["baseline", "baseline_mon", "baseline_mix"]:
            if self.type in ["baseline_mon", "baseline_mix"]:
                self.sli.setparams(self.mon, self.a, self.b, self.w)
            if self.npts < 500_000:
                self.LipsConst = self.sli.lipschitz_constant() # ~42mins for ~500K samples, provided 10^8 ops = 1s
            else:
                self.LipsConst = self.sli.lipschitz_anchor_sampling()

        if self.type in ["local", "local_mon", "local_mix"]:
            if self.type in ["local_mon", "local_mix"]:
                self.sli.setparams(self.mon, self.a, self.b, self.w)
            self.sli.compute_local_lipschitz() # no fallback method for extensive data

        # Smooth version uses the baseline_mon with reduced LipsConst
        if self.type == "smooth":
            self.sli.setparams(self.mon, self.a, self.b, self.w)
            if self.npts < 500_000:
                self.LipsConst = self.sli.lipschitz_constant() # ~42mins for ~500K samples, provided 10^8 ops = 1s
            else:
                self.LipsConst = self.sli.lipschitz_anchor_sampling()
            self.smoothened_LipConst = self.LipsConst / 2


    def predict(self):
        if self.type == "baseline":
            preds = self.sli.values(
                Q = self.X_test_np, M = self.LipsConst, model = 0, k = self.num_k
            )

        if self.type == "baseline_mon":
            preds = self.sli.values(
                Q = self.X_test_np, M = self.LipsConst, model = 1, k = self.num_k
            )

        if self.type == "baseline_mix":
            baseline_preds = self.sli.values(
                Q = self.X_test_np, M = self.LipsConst, model = 0, k = self.num_k
            )
            baseline_mon_preds = self.sli.values(
                Q = self.X_test_np, M = self.LipsConst, model = 1, k = self.num_k
            )
            preds = (baseline_preds + baseline_mon_preds) / 2

        if self.type == "local":
            preds = self.sli.values_local(
                Q = self.X_test_np, model = 0, k = self.num_k
            )

        if self.type == "local_mon":
            preds = self.sli.values_local(
                Q = self.X_test_np, model = 1, k = self.num_k
            )

        if self.type == "local_mix":
            local_preds = self.sli.values_local(
                Q = self.X_test_np, model = 0, k = self.num_k
            )
            local_mon_preds = self.sli.values_local(
                Q = self.X_test_np, model = 1, k = self.num_k
            )
            preds = (local_preds + local_mon_preds) / 2

        if self.type == "smooth":
            preds = self.sli.values(
                Q = self.X_test_np, M = self.smoothened_LipConst, model = 1, k = self.num_k
            )

        return preds
