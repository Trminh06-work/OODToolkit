from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

import json
from collections import defaultdict

from scipy.stats import friedmanchisquare, wilcoxon, rankdata


import logging

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message=r".*tensorboardX.*removed.*",
    category=UserWarning,
    module=r"pytorch_lightning.*",
)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)


class DataSaver:
    def __init__(self, model_name, results_root = "Results"):
        self.model_name = model_name
        self.output_dir = Path(results_root)
        self.output_dir.mkdir(parents = True, exist_ok = True)


    def _to_python(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if hasattr(obj, "item"):   # torch / numpy scalar
            return obj.item()
        raise TypeError


    def save_result(self, out_file, results):
        out_path = Path(out_file)
        out_path.parent.mkdir(parents = True, exist_ok = True)
        try:
            with out_path.open("w", encoding = "utf-8") as f:
                json.dump(results, f, indent = 2, default = self._to_python)
        except Exception as exc:
            raise IOError(f"Cannot save file: {out_path}") from exc


    def read_json(self, file_name):
        with open(file_name, "r", encoding = "utf-8") as f:
            data = json.load(f)
        return defaultdict(dict, data)


class AnalystModel:
    def __init__(
        self,
        alpha: float = 0.01,
        agg_method: str = "median",   # or mean (less robust)
        results_root: str = "Results",
        split_data_root: str = "../data/splitted",
        key_metric: str = "RMSE" # /"nRMSE" or "MAE"/"nMAE" or "MaxAE"/"nMaxAE",
    ):
        if key_metric not in ["RMSE", "nRMSE", "MAE", "nMAE", "MaxAE", "nMaxAE"]:
            raise ValueError('key_metric must be "RMSE", "nRMSE", "MAE", "nMAE", "MaxAE", or "nMaxAE"')
        self.alpha = alpha            # significance level
        self.agg_method = agg_method
        self.results_root = Path(results_root)
        self.split_data_root = Path(split_data_root)
        self.key_metric = key_metric


    def _resolve_results_root(self, dir_path = None):
        results_root = self.results_root if dir_path is None else Path(dir_path)
        if not results_root.exists():
            raise FileNotFoundError(f"Results directory does not exist: {results_root}")
        return results_root


    def _list_model_dirs(self, dir_path = None):
        results_root = self._resolve_results_root(dir_path)
        return sorted(path for path in results_root.iterdir() if path.is_dir())


    def _list_variants(self, model_name: str, dir_path = None):
        results_root = self._resolve_results_root(dir_path)
        model_dir = results_root / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model results directory does not exist: {model_dir}")

        variant_dirs = sorted(
            path for path in model_dir.iterdir()
            if path.is_dir() and (path / "_variant.json").exists()
        )
        if variant_dirs:
            return variant_dirs

        return [model_dir]


    def _resolve_variant_dir(self, model_name: str, variant_name: str | None = None, dir_path = None):
        variants = self._list_variants(model_name, dir_path = dir_path)
        if variant_name is None:
            for variant_dir in variants:
                if variant_dir.name == "baseline":
                    return variant_dir
            return variants[0]

        for variant_dir in variants:
            if variant_dir.name == variant_name:
                return variant_dir

        available = [path.name for path in variants]
        raise ValueError(f"Variant '{variant_name}' not found for {model_name}. Available: {available}")


    def _list_dataset_names(self, result_dir: Path):
        return sorted(
            path.stem for path in result_dir.glob("*.json")
            if path.name != "_variant.json"
        )


    def _list_split_types(self, result_dir: Path, dataset_names = None):
        dataset_names = self._list_dataset_names(result_dir) if dataset_names is None else dataset_names
        split_types = set()

        for ds_name in dataset_names:
            file_name = result_dir / f"{ds_name}.json"
            if not file_name.exists():
                continue
            with file_name.open("r", encoding = "utf-8") as f:
                res_dict = json.load(f)
            split_types.update(res_dict.keys())

        return sorted(split_types)


    def _list_model_labels(self, dir_path = None, baseline_only: bool = False, include_variants: bool = True):
        labels = []
        for model_dir in self._list_model_dirs(dir_path = dir_path):
            variants = self._list_variants(model_dir.name, dir_path = dir_path)

            if baseline_only:
                baseline_variant = next((path for path in variants if path.name == "baseline"), None)
                if baseline_variant is None or baseline_variant == model_dir:
                    labels.append(model_dir.name)
                else:
                    labels.append(f"{model_dir.name}/baseline")
                continue

            if not include_variants:
                labels.append(model_dir.name)
                continue

            for variant_dir in variants:
                if variant_dir == model_dir:
                    labels.append(model_dir.name)
                else:
                    labels.append(f"{model_dir.name}/{variant_dir.name}")
        return labels


    def _parse_model_label(self, model_label: str):
        if "/" in model_label:
            model_name, variant_name = model_label.split("/", 1)
            return model_name, variant_name
        return model_label, None


    def aggregate(self, values):
        values = np.asarray(values, dtype=float)
        if self.agg_method == "median":
            return float(np.median(values))
        elif self.agg_method == "mean":
            return float(np.mean(values))
        else:
            raise ValueError("agg_method must be 'median' or 'mean'")


    def _sort_splits(self, splits, random_split_name: str = "Random_Split"):
        splits = list(splits)
        head = [s for s in splits if s == random_split_name]
        tail = sorted(s for s in splits if s != random_split_name)
        return head + tail

    # ---------- helpers ----------
    def adaptive_format(self, x):
        """No trailing zeros + magnitude-based precision."""
        if pd.isna(x):
            return "--"
        if x < 10:
            return f"{x:.2f}".rstrip("0").rstrip(".")
        elif x < 100:
            return f"{x:.1f}".rstrip("0").rstrip(".")
        else:
            return f"{int(round(x))}"


    def split_score_by_dict(self, dict, metric, ds_name):
        """
        Input: JSON of one dataset for one model
        Output: {split_type: aggregated_metric_over_runs}
        """
        # if metric not in ["RMSE", "MAE"]:
        #     raise ValueError("metric must be 'RMSE' or 'MAE'!")

        out = {}
        for split_type, runs in dict.items():
            vals = []
            for run_idx, metrics_dict in runs.items():
                if metric in metrics_dict and metrics_dict[metric] is not None:
                    vals.append(float(metrics_dict[metric]))
            if vals:
                out[split_type] = self.aggregate(vals)
        return out


    def construct_full_stats_table(
        self,
        dir_path = None,
        metric = "RMSE", # /"nRMSE" or "MAE"/"nMAE" or "MaxAE"/"nMaxAE",
        baseline_only: bool = False,
        include_variants: bool = True,
        dataset_names = None,
    ):
        records = []
        results_root = self._resolve_results_root(dir_path)
        model_entries = []
        dataset_filter = None
        requested_dataset_set = None
        if dataset_names is not None:
            dataset_filter = list(dict.fromkeys(dataset_names))
            if not dataset_filter:
                raise ValueError("dataset_names must not be empty when provided.")
            requested_dataset_set = set(dataset_filter)

        for model_label in self._list_model_labels(
            dir_path = results_root,
            baseline_only = baseline_only,
            include_variants = include_variants,
        ):
            model_name, variant_name = self._parse_model_label(model_label)
            data_loader = DataSaver(model_name, results_root)
            save_dir = self._resolve_variant_dir(model_name, variant_name, dir_path = results_root)
            available_dataset_names = self._list_dataset_names(save_dir)

            if requested_dataset_set is None:
                selected_dataset_names = available_dataset_names
            else:
                available_dataset_set = set(available_dataset_names)
                selected_dataset_names = [
                    ds_name for ds_name in dataset_filter
                    if ds_name in available_dataset_set
                ]

            if not selected_dataset_names:
                if requested_dataset_set is None:
                    logging.warning(
                        "Skipping model '%s' because no dataset result files were found in %s",
                        model_label,
                        save_dir,
                    )
                else:
                    logging.warning(
                        "Skipping model '%s' because none of the requested datasets were found in %s: %s",
                        model_label,
                        save_dir,
                        dataset_filter,
                    )
                continue

            model_entries.append((model_label, data_loader, save_dir, selected_dataset_names))

        for model_label, data_loader, save_dir, selected_dataset_names in model_entries:
            for ds_name in selected_dataset_names:
                file_name = save_dir / f"{ds_name}.json"
                if not file_name.exists():
                    continue

                res_dict = data_loader.read_json(file_name)
                split_scores = self.split_score_by_dict(res_dict, metric, ds_name)

                for split_type, score in split_scores.items():
                    records.append({
                        "dataset": ds_name,
                        "split": split_type,
                        "model": model_label,
                        "score": score
                    })

        long_df = pd.DataFrame(records)
        if long_df.empty:
            if requested_dataset_set is None:
                raise ValueError("No data loaded. Check ROOT, folder structure, and PRIMARY_METRIC.")
            else:
                raise ValueError(
                    "No data loaded for the requested datasets. "
                    f"Requested datasets: {dataset_filter}"
                )

        return long_df

    def construct_split_agnostic_table(self, long_df: pd.DataFrame):
        """
        This framework disregards the splitting strategies for each dataset
        """
        wide = long_df.pivot_table(
            index = ["dataset", "split"],
            columns = "model",
            values = "score",
            aggfunc = "first"
        )

        # Keep only blocks where all models exist (paired comparison)
        wide = wide.dropna(axis=0, how="any")

        if wide.empty:
            raise ValueError("No complete (dataset, split) blocks where all models are present.")

        return wide


    def construct_split_wise_table(self, long_df: pd.DataFrame, split_type: str):
        """
        This framework considers splitting strategies for each dataset
        """
        available_splits = sorted(long_df["split"].unique())
        if split_type not in available_splits:
            raise ValueError(f"{split_type} does not exist. The values must be {available_splits}")

        sub = long_df[long_df["split"] == split_type]
        wide = sub.pivot_table(index = "dataset", columns = "model", values = "score", aggfunc = "first")

        # keep only datasets where ALL models exist (paired comparisons)
        wide = wide.dropna(axis=0, how="any")

        if wide.empty:
            raise ValueError(f"No complete datasets for split={split_type} across all models.")
        return wide


    def construct_dataset_wise_table(self, long_df: pd.DataFrame, dataset_name: str):
        """
        Build a per-dataset table where rows are models and columns are split regimes.
        """
        available_datasets = sorted(long_df["dataset"].unique())
        if dataset_name not in available_datasets:
            raise ValueError(f"{dataset_name} does not exist. The values must be {available_datasets}")

        sub = long_df[long_df["dataset"] == dataset_name]
        wide = sub.pivot_table(index = "model", columns = "split", values = "score", aggfunc = "first")

        # Keep only model rows where all split regimes exist (paired comparison within this dataset).
        wide = wide.dropna(axis = 0, how = "any")

        if wide.empty:
            raise ValueError(f"No complete splits for dataset={dataset_name} across all models.")
        return wide


    def construct_model_wise_table(self, long_df: pd.DataFrame, model_label: str):
        """
        This framework evaluates a single model across splitting strategies.

        The returned table keeps split regimes as columns and uses rows indexed
        by (model, dataset), so the orientation is model-first while preserving
        dataset-level paired comparison blocks.
        """
        available_models = sorted(long_df["model"].unique())
        if model_label not in available_models:
            raise ValueError(f"{model_label} does not exist. The values must be {available_models}")

        sub = long_df[long_df["model"] == model_label]
        wide = sub.pivot_table(
            index = ["model", "dataset"],
            columns = "split",
            values = "score",
            aggfunc = "first",
        )

        # Keep only paired blocks where this model has all split regimes.
        wide = wide.dropna(axis = 0, how = "any")

        if wide.empty:
            raise ValueError(f"No complete (model, dataset) blocks for model={model_label} across all splits.")
        return wide


    # Not the core test -> use to perform granular dianosistic
    def construct_model_wise_vs_random_table(
        self,
        long_df: pd.DataFrame,
        model_label: str,
        baseline_split: str = "Random_Split",
        use_relative: bool = True,
        eps: float = 1e-12,
    ):
        """
        Build a model-wise table where each split is compared to a baseline split.

        Returns a wide table indexed by dataset, with columns as non-baseline
        split regimes and values computed as either:
            score(split) - score(baseline_split), if use_relative=False
        or
            (score(split) - score(baseline_split)) / (score(baseline_split) + eps),
            if use_relative=True.

        Positive values mean the split is worse than baseline (for error metrics),
        and negative values mean better than baseline.
        """
        available_models = sorted(long_df["model"].unique())
        if model_label not in available_models:
            raise ValueError(f"{model_label} does not exist. The values must be {available_models}")

        sub = long_df[long_df["model"] == model_label]
        available_splits = sorted(sub["split"].unique())
        if baseline_split not in available_splits:
            raise ValueError(
                f"{baseline_split} does not exist for model={model_label}. "
                f"The values must be {available_splits}"
            )

        wide = sub.pivot_table(
            index = ["dataset"],
            columns = "split",
            values = "score",
            aggfunc = "first",
        )

        # Keep only paired blocks where all split regimes are present.
        wide = wide.dropna(axis = 0, how = "any")

        compare_splits = [split_type for split_type in available_splits if split_type != baseline_split]
        if not compare_splits:
            raise ValueError(
                f"No non-baseline split regimes to compare against {baseline_split} for model={model_label}."
            )

        if wide.empty:
            raise ValueError(
                f"No complete dataset blocks for model={model_label} across all splits "
                f"including baseline={baseline_split}."
            )

        baseline_vals = wide[baseline_split]
        compared = wide[compare_splits].sub(baseline_vals, axis = 0)
        if use_relative:
            compared = compared.div(baseline_vals + eps, axis = 0)
        return compared


    def construct_model_robustness_table(
        self,
        long_df: pd.DataFrame,
        model_label: str,
        baseline_split: str = "Random_Split",
        eps: float = 1e-12,
    ):
        """
        Returns:
            - relative degradation table (Δ_rel)
            - worst-case degradation per dataset
            - robustness score per model
        """

        available_models = sorted(long_df["model"].unique())
        if model_label not in available_models:
            raise ValueError(f"{model_label} not found. Available: {available_models}")

        sub = long_df[long_df["model"] == model_label]

        wide = sub.pivot_table(
            index=["dataset"],
            columns="split",
            values="score",
            aggfunc="first",
        )

        # keep only complete rows
        wide = wide.dropna(axis=0, how="any")

        if baseline_split not in wide.columns:
            raise ValueError(f"{baseline_split} not found in splits")

        baseline = wide[baseline_split]

        compare_splits = [s for s in wide.columns if s != baseline_split]

        # --- FIX 1: relative degradation ---
        # epsilon keeps the denominator non-zero in degenerate cases.
        delta_rel = wide[compare_splits].sub(baseline, axis=0).div(baseline + eps, axis=0)

        # --- FIX 2: worst-case per dataset ---
        worst_case = delta_rel.max(axis=1)

        # --- FIX 3: robustness score ---
        robustness_score = delta_rel.max(axis = 1).mean()   # average worst-case per dataset

        return delta_rel, worst_case, robustness_score


    # ---------------------------------------------------------------------------------------------
    # Hypothesis Testing Framework - Friedman's test and Post-hoc Analysis
    # ---------------------------------------------------------------------------------------------
    def friedman_on_wide(self, wide: pd.DataFrame):
        arrays = [wide[c].to_numpy(dtype=float) for c in wide.columns]
        return friedmanchisquare(*arrays)


    def compute_mean_ranks(self, wide: pd.DataFrame):
        ranks = wide.apply(
            lambda row: pd.Series(rankdata(row.to_numpy(), method="average"), index = wide.columns),
            axis = 1
        )
        mean_ranks = ranks.mean(axis=0).sort_values()
        return mean_ranks


    def posthoc_vs_best(self, wide: pd.DataFrame, best: str):
        pvals, comps, effects = [], [], []
        x = wide[best].to_numpy(dtype=float)

        for m in wide.columns:
            if m == best:
                continue
            y = wide[m].to_numpy(dtype=float)
            _, p = wilcoxon(x, y, alternative="less")  # best < other
            pvals.append(float(p))
            comps.append(m)
            effects.append(float(np.median(wide[m] - wide[best])))  # positive => best better

        p_holm = self.holm_adjust(pvals)

        post = pd.DataFrame({
            "compare_to": comps,
            "p_value": pvals,
            "p_holm": p_holm,
            "median(other - best)": effects
        }).sort_values("p_holm")

        return post


    def holm_adjust(self, pvals):
        """
        Holm-Bonferroni adjusted p-values (step-down), returns adjusted p-values.
        """
        pvals = np.array(pvals, dtype=float)
        m = len(pvals)
        order = np.argsort(pvals)
        adj = np.empty(m, dtype=float)
        running_max = 0.0
        for k, idx in enumerate(order):
            val = (m - k) * pvals[idx]
            running_max = max(running_max, val)
            adj[idx] = min(running_max, 1.0)
        return adj.tolist()


    # ---------------------------------------------------------------------------------------------
    # (LaTeX formatting)
    # ---------------------------------------------------------------------------------------------
    def _fmt_rank_latex(self, val: float, tag: str) -> str:
        """tag in {'best','tie','normal'}."""
        if tag == "best":
            return rf"\textcolor{{red}}{{{val:.2f}}}"
        if tag == "tie":
            return rf"\textcolor{{blue}}{{{val:.2f}}}"
        return f"{val:.2f}"


    # Benchmark all methods against the baseline method
    def _fmt_delta_latex(self, val: float) -> str:
        """Format split-baseline delta for LaTeX (green=better, red=worse)."""
        if pd.isna(val):
            return "--"

        txt = f"{val:+.3f}".rstrip("0").rstrip(".")
        if txt in {"+0", "-0"}:
            txt = "0"

        if val < 0:
            return rf"\textcolor{{teal}}{{{txt}}}"
        if val > 0:
            return rf"\textcolor{{red}}{{{txt}}}"
        return txt


    def _fmt_nrmse_latex(self, val: float) -> str:
        if pd.isna(val):
            return "--"
        return self.adaptive_format(float(val))


    def print_nrmse_table_latex(self, wide: pd.DataFrame, row_header: str) -> None:
        """
        Print a full numeric table as LaTeX rows.
        """
        header_cells = [row_header] + [str(col).replace("_", r"\_") for col in wide.columns]
        print(" & ".join(header_cells) + r" \\")
        print()

        for idx, row in wide.iterrows():
            row_name = str(idx).replace("_", r"\_")
            value_cells = []
            for val in row.tolist():
                txt = self._fmt_nrmse_latex(val)
                value_cells.append("--" if txt == "--" else f"${txt}$")
            print(" & ".join([row_name] + value_cells) + r" \\")
            print()


    def print_splitwise_meanrank_latex(self, latex_buffer: dict, split_types) -> None:
        """
        latex_buffer: {model: {split_type: (rank, tag)}}
        Prints rows like: Model & 9.15 & ... \\
        """
        for model in sorted(latex_buffer.keys()):
            if model not in latex_buffer:
                raise ValueError("model does not exist")
            # Escape underscores for LaTeX
            model_name = str(model).replace("_", r"\_")
            row = [model_name]
            for s in split_types:
                if s not in latex_buffer[model]:
                    row.append("--")
                    continue
                rank, tag = latex_buffer[model][s]
                row.append(self._fmt_rank_latex(float(rank), tag))
            print(" & ".join(row) + r" \\")
            print()  # blank line between models


    def print_modelwise_meanrank_latex(self, latex_buffer: dict, model_labels) -> None:
        """
        latex_buffer: {split_type: {model_label: (rank, tag)}}
        Prints rows like: Split & 2.10 & ... \\
        """
        for split_type in sorted(latex_buffer.keys()):
            split_name = str(split_type).replace("_", r"\_")
            row = [split_name]
            for model_label in model_labels:
                if model_label not in latex_buffer[split_type]:
                    row.append("--")
                    continue
                rank, tag = latex_buffer[split_type][model_label]
                row.append(self._fmt_rank_latex(float(rank), tag))
            print(" & ".join(row) + r" \\")
            print()


    def print_modelwise_vs_random_latex(self, latex_buffer: dict, model_labels, split_types) -> None:
        """
        latex_buffer: {model_label: {split_type: delta_vs_random}}
        Prints rows like: Model & -0.12 & +0.03 & ... \\
        """
        for model_label in model_labels:
            model_name = str(model_label).replace("_", r"\_")
            row = [model_name]
            for split_type in split_types:
                if model_label not in latex_buffer or split_type not in latex_buffer[model_label]:
                    row.append("--")
                    continue
                row.append(self._fmt_delta_latex(float(latex_buffer[model_label][split_type])))
            print(" & ".join(row) + r" \\")
            print()


    # ---------------------------------------------------------------------------------------------
    # Hypothesis Tests
    # ---------------------------------------------------------------------------------------------

    def split_agnostic_test(self, long_df: pd.DataFrame = None, dataset_names = None):
        """
        This framework disregards the splitting strategies, exclude Random Split due to iid behaviour, for each dataset

        Null hypothesis: All models perform equally
        Alt hypothesis : At least 1 model performs significantly different
        """
        if long_df is None:
            long_df = self.construct_full_stats_table(
                metric = self.key_metric,
                # baseline_only = True,
                dataset_names = dataset_names,
            )

        # Exclude Random_Split rows
        long_df = long_df[long_df["split"] != "Random_Split"].copy()

        wide = self.construct_split_agnostic_table(long_df)

        chi2, p_friedman = self.friedman_on_wide(wide)
        print(f"Friedman chi2 = {chi2:.4f}, p={p_friedman:.6g}")

        # Find the best candidate by mean rank
        mean_ranks = self.compute_mean_ranks(wide)
        best = mean_ranks.index[0]
        print("Mean ranks (lower is better):")
        print(mean_ranks)

        # Post-hoc Holm-corrected Wilcoxon vs best
        # H0: median(X_best - X_others) = 0
        # HA: X_best < X_others
        if p_friedman < self.alpha:
            posthoc = self.posthoc_vs_best(wide, best)
            print("\nPost-hoc (Holm-corrected Wilcoxon vs best):")
            print(posthoc.to_string(index=False))

            top_group = [best] + posthoc.loc[posthoc["p_holm"] >= self.alpha, "compare_to"].tolist()

            if len(top_group) == 1:
                print(f"\n✅ Best overall model (split-agnostic): {best}")
            else:
                print(f"\n⚠️ No single winner. Top group (ties with {best}): {top_group}")
        else:
            print(f"\n⚠️ Friedman not significant (p≥{self.alpha})")


    def split_wise_test(
        self,
        long_df: pd.DataFrame = None,
        baseline_only: bool = True,
        include_variants: bool = False,
        dataset_names = None,
    ):
        if long_df is None:
            long_df = self.construct_full_stats_table(
                metric = self.key_metric,
                baseline_only = baseline_only,
                include_variants = include_variants,
                dataset_names = dataset_names,
            )

        summary_rows = []

        # buffer to build LaTeX rows at the end
        latex_buffer = {}  # {model: {split_type: (mean_rank, tag)}}

        split_types = self._sort_splits(long_df["split"].unique())

        for split_type in split_types:
            print("\n" + "=" * 90)
            print(f"SPLIT: {split_type}")

            # Step 3: wide matrix for this split
            wide = self.construct_split_wise_table(long_df, split_type)
            print(f"Datasets (paired blocks): {wide.shape[0]} | Models: {wide.shape[1]}")

            # Step 4: Friedman
            chi2, p_friedman = self.friedman_on_wide(wide)
            print(f"Friedman chi2={chi2:.4f}, p={p_friedman:.6g}")

            # Step 5: mean ranks
            mr = self.compute_mean_ranks(wide)
            best = mr.index[0]
            print("\nMean ranks (lower is better):")
            print(mr)
            print("\nCandidate best:", best)

            # NEW: default highlight tags for LaTeX
            highlight = {m: "normal" for m in mr.index}
            highlight[best] = "best"

            # Step 6: post-hoc
            if p_friedman < self.alpha and wide.shape[1] > 1:
                post = self.posthoc_vs_best(wide, best)
                print("\nPost-hoc (Holm-corrected Wilcoxon vs best):")
                print(post.to_string(index=False))

                top_group = [best] + post.loc[post["p_holm"] >= self.alpha, "compare_to"].tolist()

                # mark ties (incl best) in blue, best in red
                for m in top_group:
                    highlight[m] = "tie"
                highlight[best] = "best"

                if len(top_group) == 1:
                    conclusion = f"BEST: {best}"
                    print(f"\n✅ Best model under split '{split_type}': {best}")
                else:
                    conclusion = f"TOP_GROUP: {top_group}"
                    print(f"\n⚠️ No single winner under split '{split_type}'. Top group: {top_group}")
            else:
                post = None
                conclusion = "NO_SIG_DIFF"
                print(f"\n⚠️ No significant overall difference under split '{split_type}' (or only 1 model).")

            # store mean ranks + highlight tags for LaTeX export
            for model, rank in mr.items():
                latex_buffer.setdefault(model, {})
                latex_buffer[model][split_type] = (float(rank), highlight[model])

            summary_rows.append({
                "split": split_type,
                "datasets_used": wide.shape[0],
                "models": wide.shape[1],
                "friedman_p": p_friedman,
                "best_candidate": best,
                "conclusion": conclusion
            })

        summary = pd.DataFrame(summary_rows).sort_values("split")
        print("\n" + "#" * 90)
        print("SPLIT-WISE SUMMARY")
        print(summary.to_string(index=False))

        # print LaTeX rows in your requested style
        print("The split types are:")
        for split in split_types:
            print(split, end = " | ")

        print("\n" + "#" * 90)
        print("LATEX ROWS (mean ranks per split; red=best, blue=tied-with-best)\n")
        self.print_splitwise_meanrank_latex(latex_buffer, split_types)


    def per_dataset_table_test(
        self,
        long_df: pd.DataFrame = None,
        baseline_only: bool = True,
        include_variants: bool = False,
        print_latex: bool = True,
        dataset_names = None,
    ):
        """
        Print one full 2D nRMSE table per dataset.
        Table orientation: rows=models, columns=splits.
        """
        if long_df is None:
            long_df = self.construct_full_stats_table(
                metric = self.key_metric,
                baseline_only = baseline_only,
                include_variants = include_variants,
                dataset_names = dataset_names,
            )

        summary_rows = []
        dataset_names = sorted(long_df["dataset"].unique())

        for dataset_name in dataset_names:
            print("\n" + "=" * 90)
            print(f"DATASET: {dataset_name}")

            wide = self.construct_dataset_wise_table(long_df, dataset_name)
            wide = wide.reindex(index = sorted(wide.index), columns = self._sort_splits(wide.columns))

            print(f"Models (complete): {wide.shape[0]} | Splits: {wide.shape[1]}")
            print(f"\nRaw {self.key_metric} table (rows=models, cols=splits):")
            print(wide.to_string())

            if print_latex:
                print("\n" + "#" * 90)
                print(f"LATEX ROWS (dataset={dataset_name}; rows=models, cols=splits)\n")
                self.print_nrmse_table_latex(wide, row_header = "Model")

            summary_rows.append({
                "dataset": dataset_name,
                "models": wide.shape[0],
                "splits": wide.shape[1],
            })

        summary = pd.DataFrame(summary_rows)
        print("\n" + "#" * 90)
        print("PER-DATASET TABLE SUMMARY")
        print(summary.to_string(index=False))


    def model_wise_test(
        self,
        long_df: pd.DataFrame = None,
        baseline_only: bool = True,
        include_variants: bool = False,
        dataset_names = None,
    ):
        if long_df is None:
            long_df = self.construct_full_stats_table(
                metric = self.key_metric,
                baseline_only = baseline_only,
                include_variants = include_variants,
                dataset_names = dataset_names,
            )

        summary_rows = []

        # Build per-model for model-row/split-column LaTeX output.
        latex_buffer_by_model = {}  # {model_label: {split_type: (mean_rank, tag)}}
        split_types_seen = set()

        model_labels = sorted(long_df["model"].unique())

        for model_label in model_labels:
            print("\n" + "=" * 90)
            print(f"MODEL: {model_label}")

            # Step 3: wide matrix for this model
            wide = self.construct_model_wise_table(long_df, model_label)
            print(f"Datasets (paired blocks): {wide.shape[0]} | Splits: {wide.shape[1]}")

            # Step 4: Friedman
            chi2, p_friedman = self.friedman_on_wide(wide)
            print(f"Friedman chi2={chi2:.4f}, p={p_friedman:.6g}")

            # Step 5: mean ranks
            mr = self.compute_mean_ranks(wide)
            best = mr.index[0]
            print("\nMean ranks (lower is better):")
            print(mr)
            print("\nCandidate best split:", best)

            highlight = {split_type: "normal" for split_type in mr.index}
            highlight[best] = "best"

            # Step 6: post-hoc
            if p_friedman < self.alpha and wide.shape[1] > 1:
                post = self.posthoc_vs_best(wide, best)
                print("\nPost-hoc (Holm-corrected Wilcoxon vs best):")
                print(post.to_string(index=False))

                top_group = [best] + post.loc[post["p_holm"] >= self.alpha, "compare_to"].tolist()

                for split_type in top_group:
                    highlight[split_type] = "tie"
                highlight[best] = "best"

                if len(top_group) == 1:
                    conclusion = f"BEST: {best}"
                    print(f"\n✅ Best split for model '{model_label}': {best}")
                else:
                    conclusion = f"TOP_GROUP: {top_group}"
                    print(f"\n⚠️ No single winner for model '{model_label}'. Top group: {top_group}")
            else:
                post = None
                conclusion = "NO_SIG_DIFF"
                print(f"\n⚠️ No significant overall difference across splits for model '{model_label}' (or only 1 split).")

            # store mean ranks + highlight tags for LaTeX export
            for split_type, rank in mr.items():
                split_types_seen.add(split_type)
                latex_buffer_by_model.setdefault(model_label, {})
                latex_buffer_by_model[model_label][split_type] = (float(rank), highlight[split_type])

            summary_rows.append({
                "model": model_label,
                "datasets_used": wide.shape[0],
                "splits": wide.shape[1],
                "friedman_p": p_friedman,
                "best_candidate": best,
                "conclusion": conclusion
            })

        summary = pd.DataFrame(summary_rows).sort_values("model")
        print("\n" + "#" * 90)
        print("MODEL-WISE SUMMARY")
        print(summary.to_string(index=False))

        split_types = self._sort_splits(split_types_seen)

        print("The split types are:")
        for split in split_types:
            print(split, end = " | ")

        print("\n" + "#" * 90)
        print("LATEX ROWS (rows=models, cols=splits; red=best, blue=tied-with-best)\n")
        self.print_splitwise_meanrank_latex(latex_buffer_by_model, split_types)


    def robustness_model_comparison_latex(
        self,
        long_df: pd.DataFrame = None,
        baseline_only: bool = True,
        include_variants: bool = False,
        baseline_split: str = "Random_Split",
        dataset_names = None,
    ):
        """
        Robustness comparison across models with LaTeX output.
        Uses relative degradation Δ_rel = (S - R) / R
        """
        if long_df is None:
            long_df = self.construct_full_stats_table(
                metric = self.key_metric,
                baseline_only = baseline_only,
                include_variants = include_variants,
                dataset_names = dataset_names,
            )

        models = sorted(long_df["model"].unique())

        model_tables = []
        scores = {}

        # ---------- compute Δ_rel ----------
        for model in models:
            delta_rel, _, score = self.construct_model_robustness_table(long_df, model)
            scores[model] = score

            tmp = delta_rel.copy()
            tmp["model"] = model
            model_tables.append(tmp)

        combined = pd.concat(model_tables)

        # reshape
        melted = combined.reset_index().melt(
            id_vars=["dataset", "model"],
            var_name="split",
            value_name="delta_rel"
        )

        wide = melted.pivot_table(
            index=["dataset", "split"],
            columns="model",
            values="delta_rel",
            aggfunc="first"
        ).dropna()

        if wide.empty:
            raise ValueError("No complete paired blocks for robustness test.")

        common_index = wide.index

        scores = {}
        for model in wide.columns:
            vals = wide[model].loc[common_index]
            scores[model] = float(vals.mean())

        # ---------- Friedman ----------
        arrays = [wide[c].values for c in wide.columns]
        chi2, p = friedmanchisquare(*arrays)

        # ---------- ranking ----------
        ranks = wide.apply(
            lambda row: pd.Series(rankdata(row.values, method="average"), index=wide.columns),
            axis=1
        )
        mean_ranks = ranks.mean().sort_values()

        best = mean_ranks.index[0]

        # ---------- posthoc ----------
        if p < self.alpha:
            posthoc = self.posthoc_vs_best(wide, best)

            # models NOT significantly worse than best
            top_group = [best] + posthoc.loc[
                posthoc["p_holm"] >= self.alpha, "compare_to"
            ].tolist()
        else:
            # no significance → all tied
            top_group = list(mean_ranks.index)

        # ---------- LaTeX formatting ----------
        latex_rows = []

        for i, model in enumerate(mean_ranks.index):
            rank_val = mean_ranks[model]
            score_val = scores[model]

            # formatting
            rank_txt = f"{rank_val:.2f}"
            score_txt = f"{score_val:.4f}"

            if model == best:
                rank_txt = f"\\textcolor{{red}}{{{rank_txt}}}"
            elif model in top_group:
                rank_txt = f"\\textcolor{{blue}}{{{rank_txt}}}"

            row = f"{model} & ${rank_txt}$ & ${score_txt}$ \\\\"
            latex_rows.append(row)

        # ---------- print ----------
        print("%% Robustness Model Comparison (LaTeX)")
        print(f"%% Friedman chi2 = {chi2:.4f}, p = {p:.6g}")
        print("\n".join(latex_rows))


    # Not the core test -> use to perform granular dianosistic
    def model_wise_vs_random_latex_table(
        self,
        long_df: pd.DataFrame = None,
        baseline_only: bool = True,
        include_variants: bool = False,
        baseline_split: str = "Random_Split",
        dataset_names = None,
    ):
        """
        Build model-wise split deltas versus Random_Split and print LaTeX rows.

        Each cell is:
            aggregate_over_datasets((score(split) - score(baseline_split)) / (score(baseline_split) + eps))
        where aggregate_over_datasets follows self.agg_method.
        """
        if long_df is None:
            long_df = self.construct_full_stats_table(
                metric = self.key_metric,
                baseline_only = baseline_only,
                include_variants = include_variants,
                dataset_names = dataset_names,
            )

        available_splits = sorted(long_df["split"].unique())
        if baseline_split not in available_splits:
            raise ValueError(f"{baseline_split} does not exist. The values must be {available_splits}")

        split_types = [split_type for split_type in available_splits if split_type != baseline_split]
        if not split_types:
            raise ValueError(f"No non-baseline split regimes to compare against {baseline_split}.")

        model_labels = sorted(long_df["model"].unique())
        latex_buffer = {}  # {model_label: {split_type: delta_rel}}
        summary_rows = []
        eps = 1e-12

        for model_label in model_labels:
            print("\n" + "=" * 90)
            print(f"MODEL: {model_label}")

            wide_delta = self.construct_model_wise_vs_random_table(
                long_df,
                model_label,
                baseline_split = baseline_split,
                use_relative = True,
                eps = eps,
            )
            print(f"Datasets (paired blocks): {wide_delta.shape[0]} | Compared splits: {wide_delta.shape[1]}")

            agg_delta = wide_delta.apply(
                lambda col: self.aggregate(col.to_numpy(dtype=float)),
                axis = 0,
            ).sort_values()

            print(
                f"\nAggregated relative delta vs {baseline_split} "
                f"((split - baseline)/(baseline + {eps:.0e}); lower is better):"
            )
            print(agg_delta)

            for split_type, delta in agg_delta.items():
                latex_buffer.setdefault(model_label, {})
                latex_buffer[model_label][split_type] = float(delta)

            summary_rows.append({
                "model": model_label,
                "datasets_used": wide_delta.shape[0],
                "splits_compared": wide_delta.shape[1],
                "best_split_vs_random": agg_delta.index[0],
                "best_delta_rel": float(agg_delta.iloc[0]),
            })

        summary = pd.DataFrame(summary_rows).sort_values("model")
        print("\n" + "#" * 90)
        print("MODEL-WISE VS RANDOM SUMMARY")
        print(summary.to_string(index=False))

        print("The split types are:")
        for split in split_types:
            print(split, end = " | ")

        print("\n" + "#" * 90)
        print(f"LATEX ROWS (delta_rel vs {baseline_split}; teal=better, red=worse)\n")
        self.print_modelwise_vs_random_latex(latex_buffer, model_labels, split_types)


    # ---------------------------------------------------------------------------------------------
    # Runtime Performance Framework - training/inference time comparison
    # ---------------------------------------------------------------------------------------------
    # Column label for the derived per-query inference time.
    PER_QUERY_COL = "Inference (per query)"

    def construct_runtime_long_table(
        self,
        dir_path = None,
        baseline_only: bool = True,
        include_variants: bool = False,
        dataset_names = None,
        time_metrics = ("training_time", "inference_time"),
    ):
        """
        Build a per-run runtime table with one row per (dataset, split, model, run).

        Unlike the error-metric tables, runs are NOT collapsed here: keeping the raw
        per-run values (as stored in the result JSON) lets the summary report a mean
        and a standard deviation. Returns a DataFrame with columns:
        dataset, split, model, run, <time_metrics...>.
        """
        results_root = self._resolve_results_root(dir_path)

        dataset_filter = None
        if dataset_names is not None:
            dataset_filter = list(dict.fromkeys(dataset_names))
            if not dataset_filter:
                raise ValueError("dataset_names must not be empty when provided.")
        dataset_filter_set = set(dataset_filter) if dataset_filter is not None else None

        records = []
        for model_label in self._list_model_labels(
            dir_path = results_root,
            baseline_only = baseline_only,
            include_variants = include_variants,
        ):
            model_name, variant_name = self._parse_model_label(model_label)
            data_loader = DataSaver(model_name, results_root)
            save_dir = self._resolve_variant_dir(model_name, variant_name, dir_path = results_root)
            available_dataset_names = self._list_dataset_names(save_dir)

            if dataset_filter_set is None:
                selected_dataset_names = available_dataset_names
            else:
                available_dataset_set = set(available_dataset_names)
                selected_dataset_names = [
                    ds_name for ds_name in dataset_filter
                    if ds_name in available_dataset_set
                ]

            for ds_name in selected_dataset_names:
                file_name = save_dir / f"{ds_name}.json"
                if not file_name.exists():
                    continue

                res_dict = data_loader.read_json(file_name)
                for split_type, runs in res_dict.items():
                    for run_idx, metrics_dict in runs.items():
                        row = {
                            "dataset": ds_name,
                            "split": split_type,
                            "model": model_label,
                            "run": run_idx,
                        }
                        has_value = False
                        for time_metric in time_metrics:
                            val = metrics_dict.get(time_metric)
                            if val is not None:
                                row[time_metric] = float(val)
                                has_value = True
                        if has_value:
                            records.append(row)

        long_df = pd.DataFrame(records)
        if long_df.empty:
            raise ValueError(
                "No runtime data loaded. Check results root, folder structure, and time_metrics."
            )
        return long_df


    def construct_runtime_summary_table(
        self,
        runtime_long_df: pd.DataFrame = None,
        time_metrics = ("training_time", "inference_time"),
        n_samples: int = 30000,
        per_query_metric: str = "inference_time",
        baseline_only: bool = True,
        include_variants: bool = False,
        dataset_names = None,
    ):
        """
        Per-model runtime summary: rows = models, columns = metrics.

        Runtime tables always aggregate the pooled per-run measurements with the
        MEAN (independent of self.agg_method) and report the standard deviation.
        A derived "Inference (per query)" column is added as per_query_metric divided
        by n_samples (e.g. 30k samples in the current experiment).

        Returns a DataFrame indexed by model with a column MultiIndex (metric, stat)
        where stat is in {"mean", "std"}.
        """
        if runtime_long_df is None:
            runtime_long_df = self.construct_runtime_long_table(
                baseline_only = baseline_only,
                include_variants = include_variants,
                dataset_names = dataset_names,
                time_metrics = time_metrics,
            )

        metrics = list(time_metrics)
        grouped = runtime_long_df.groupby("model")
        mean_df = grouped[metrics].mean()
        std_df = grouped[metrics].std(ddof = 1).fillna(0.0)

        if n_samples and per_query_metric in metrics:
            mean_df[self.PER_QUERY_COL] = mean_df[per_query_metric] / n_samples
            std_df[self.PER_QUERY_COL] = std_df[per_query_metric] / n_samples
            metrics = metrics + [self.PER_QUERY_COL]

        summary = pd.concat({"mean": mean_df, "std": std_df}, axis = 1)
        # Reorder to (metric, stat) with metrics in their logical order.
        summary = summary.swaplevel(axis = 1)[metrics]
        return summary


    def _num_to_latex(self, val: float) -> str:
        """Format a number to 4 sig figs, rendering 1.2e-07 as 1.2 \\times 10^{-7}."""
        txt = f"{val:.4g}"
        if "e" in txt.lower():
            mantissa, exp = txt.lower().split("e")
            return rf"{mantissa} \times 10^{{{int(exp)}}}"
        return txt


    def _fmt_runtime_latex(self, mean: float, std: float = None, is_best: bool = False) -> str:
        """Format a runtime cell as 'mean ± std' for LaTeX; red marks the fastest model."""
        if pd.isna(mean):
            return "--"
        mean_txt = self._num_to_latex(mean)
        if is_best:
            mean_txt = rf"\textcolor{{red}}{{{mean_txt}}}"
        if std is None or pd.isna(std):
            return mean_txt
        return rf"{mean_txt} \pm {self._num_to_latex(std)}"


    def runtime_comparison_latex(
        self,
        runtime_long_df: pd.DataFrame = None,
        baseline_only: bool = True,
        include_variants: bool = False,
        dataset_names = None,
        time_metrics = ("training_time", "inference_time"),
        n_samples: int = 30000,
        per_query_metric: str = "inference_time",
        rank_by: str = "inference_time",
    ):
        """
        Print a runtime comparison table (rows = models, cols = metrics) and the
        matching LaTeX rows. Each cell is 'mean ± std' (mean aggregation), with the
        fastest model per column in red. A derived "Inference (per query)" column
        (per_query_metric / n_samples) is appended.
        """
        if rank_by not in time_metrics:
            raise ValueError(f"rank_by must be one of {list(time_metrics)}")

        if runtime_long_df is None:
            runtime_long_df = self.construct_runtime_long_table(
                baseline_only = baseline_only,
                include_variants = include_variants,
                dataset_names = dataset_names,
                time_metrics = time_metrics,
            )

        summary = self.construct_runtime_summary_table(
            runtime_long_df,
            time_metrics = time_metrics,
            n_samples = n_samples,
            per_query_metric = per_query_metric,
        )
        summary = summary.sort_values((rank_by, "mean"))

        # Display metrics in column order (time metrics + derived per-query column).
        display_metrics = list(dict.fromkeys(summary.columns.get_level_values(0)))

        print("\n" + "=" * 90)
        print(f"RUNTIME COMPARISON (agg=mean ± std; seconds; n_samples={n_samples}; ranked by {rank_by})")
        print(f"Models: {summary.shape[0]} | Metrics: {display_metrics}")
        print("\nRaw runtime table (rows=models, cols=(metric, stat)):")
        print(summary.to_string())

        # Fastest (lowest mean) model per metric.
        best_per_metric = {metric: summary[(metric, "mean")].idxmin() for metric in display_metrics}

        print("\n" + "#" * 90)
        print("LATEX ROWS (runtime in seconds, mean ± std; red=fastest per column)\n")

        header_cells = ["Model"] + [str(metric).replace("_", r"\_") for metric in display_metrics]
        print(" & ".join(header_cells) + r" \\")
        print()

        for model in summary.index:
            model_name = str(model).replace("_", r"\_")
            value_cells = []
            for metric in display_metrics:
                txt = self._fmt_runtime_latex(
                    summary.loc[model, (metric, "mean")],
                    summary.loc[model, (metric, "std")],
                    is_best = (best_per_metric[metric] == model),
                )
                value_cells.append("--" if txt == "--" else f"${txt}$")
            print(" & ".join([model_name] + value_cells) + r" \\")
            print()

        return summary
