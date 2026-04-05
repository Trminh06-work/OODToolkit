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
        alpha: float = 0.05,
        agg_method: str = "median",   # or mean (less robust)
        results_root: str = "Results",
        split_data_root: str = "../data/splitted",
    ):
        self.alpha = alpha            # significance level
        self.agg_method = agg_method
        self.results_root = Path(results_root)
        self.split_data_root = Path(split_data_root)


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


    def compute_ds_std(self, train_file):
        df_train = pd.read_parquet(train_file)

        target = df_train.iloc[:, -1]
        return float(np.std(target, ddof = 0) + 0.0001) # avoid division by zero


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


    def colorize_cell(self, cell_txt: str, is_row_best: bool, is_col_best: bool) -> str:
        if cell_txt == "--":
            return cell_txt
        if is_row_best and is_col_best:
            return f"\\textcolor{{green}}{{{cell_txt}}}"
        elif is_row_best:
            return f"\\textcolor{{blue}}{{{cell_txt}}}"
        elif is_col_best:
            return f"\\textcolor{{red}}{{{cell_txt}}}"
        else:
            return cell_txt


    def build_wide_numeric(self, metric: str, data_loader: DataSaver, save_dir: Path, model_label: str) -> pd.DataFrame:
        records = []
        dataset_names = self._list_dataset_names(save_dir)
        split_types = self._list_split_types(save_dir, dataset_names)

        for ds_name in dataset_names:
            file_name = save_dir / f"{ds_name}.json"
            if not file_name.exists():
                continue

            res_dict = data_loader.read_json(file_name)
            split_scores = self.split_score_by_dict(res_dict, metric, ds_name)

            for split_type, score in split_scores.items():
                # keep your sMAPE scaling convention
                if metric == "sMAPE" and score is not None:
                    score = score / 100.0
                records.append({"dataset": ds_name, "split": split_type, "score": score})

        long_df = pd.DataFrame(records)
        if long_df.empty:
            raise ValueError(f"No data loaded for model={model_label}, metric={metric}.")

        wide = long_df.pivot_table(
            index = "dataset",
            columns = "split",
            values = "score",
            aggfunc = "first"
        ).reindex(index = dataset_names, columns = split_types)

        return wide


    def performance_table(self, model_name, variant_name: str = None):
        result_dir = self._resolve_variant_dir(model_name, variant_name)
        model_label = model_name if variant_name is None and result_dir.name == model_name else f"{model_name}/{result_dir.name}"
        data_loader = DataSaver(model_name, self.results_root)

        # ---------- numeric tables ----------
        rmse_num = self.build_wide_numeric("RMSE", data_loader, result_dir, model_label)
        mae_num  = self.build_wide_numeric("MAE", data_loader, result_dir, model_label)
        smape_num = self.build_wide_numeric("sMAPE", data_loader, result_dir, model_label)

        # ---------- RMSE-based minima for coloring ----------
        rmse_row_min = rmse_num.min(axis = 1, skipna = True)
        global_best_rmse = rmse_num.min(axis = 1, skipna = True).min()

        # ---------- generate LaTeX rows: each cell = nRMSE / nMAE / sMAPE, colored ONLY by RMSE ----------
        latex_rows = []
        for i, ds in enumerate(rmse_num.index):
            row_cells = [f"\\#{i+1}\n"]

            for split in rmse_num.columns:
                r = rmse_num.loc[ds, split]
                m = mae_num.loc[ds, split]
                s = smape_num.loc[ds, split]

                # If any metric missing, show "-- / -- / --" (still color by RMSE if RMSE exists)
                r_txt = self.adaptive_format(r)
                m_txt = self.adaptive_format(m)
                s_txt = self.adaptive_format(s)

                cell_plain = f"{r_txt} / {m_txt} / {s_txt}"

                # highlight the best split within each dataset; mark the single global best separately
                is_row_best = (r_txt != "--") and (r == rmse_row_min.loc[ds])
                is_col_best = (r_txt != "--") and (r == global_best_rmse)

                cell_colored = self.colorize_cell(cell_plain, is_row_best, is_col_best)

                row_cells.append(f"${cell_colored}$")

            latex_rows.append(" & ".join(row_cells) + " \\\\")
        print("\n\n".join(latex_rows))

    def side_exp_performance_table(
        self,
        model_name,
        baseline_variant: str = "baseline",
        compare_variant: str = None,
    ):
        if compare_variant is None:
            raise ValueError("compare_variant must be provided")

        data_loader = DataSaver(model_name, self.results_root)
        base_dir = self._resolve_variant_dir(model_name, baseline_variant)
        compare_dir = self._resolve_variant_dir(model_name, compare_variant)

        base_rmse_num = self.build_wide_numeric("RMSE", data_loader, base_dir, f"{model_name}/{base_dir.name}")
        rmse_num = self.build_wide_numeric("RMSE", data_loader, compare_dir, f"{model_name}/{compare_dir.name}")
        mae_num  = self.build_wide_numeric("MAE", data_loader, compare_dir, f"{model_name}/{compare_dir.name}")
        smape_num = self.build_wide_numeric("sMAPE", data_loader, compare_dir, f"{model_name}/{compare_dir.name}")

        common_datasets = rmse_num.index.intersection(base_rmse_num.index)
        common_splits = rmse_num.columns.intersection(base_rmse_num.columns)
        if len(common_datasets) == 0 or len(common_splits) == 0:
            raise ValueError(
                f"No common dataset/split coverage between {model_name}/{base_dir.name} and {model_name}/{compare_dir.name}"
            )

        base_rmse_num = base_rmse_num.loc[common_datasets, common_splits]
        rmse_num = rmse_num.loc[common_datasets, common_splits]
        mae_num = mae_num.loc[common_datasets, common_splits]
        smape_num = smape_num.loc[common_datasets, common_splits]

        latex_rows = []
        for i, ds in enumerate(rmse_num.index):
            row_cells = [f"\\#{i+1}\n"]

            for split in rmse_num.columns:
                r = rmse_num.loc[ds, split]
                base_r = base_rmse_num.loc[ds, split]
                m = mae_num.loc[ds, split]
                s = smape_num.loc[ds, split]

                r_txt = self.adaptive_format(r)
                m_txt = self.adaptive_format(m)
                s_txt = self.adaptive_format(s)

                if pd.isna(base_r) or base_r == 0:
                    diff_txt = "NA"
                    diff_value = None
                else:
                    diff_value = ((r - base_r) / base_r) * 100
                    diff_txt = f"{diff_value:.1f}\\%"

                cell_plain = f"{r_txt} / {m_txt} / {s_txt} ({diff_txt})"

                if diff_value is None or diff_value == 0:
                    cell_colored = f"\\textcolor{{black}}{{{cell_plain}}}"
                elif diff_value < 0:
                    cell_colored = f"\\textcolor{{ForestGreen}}{{{cell_plain}}}"
                else:
                    cell_colored = f"\\textcolor{{BrickRed}}{{{cell_plain}}}"

                row_cells.append(f"${cell_colored}$")

            latex_rows.append(" & ".join(row_cells) + " \\\\")
        print("\n\n".join(latex_rows))


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
                    train_file = self.split_data_root / ds_name / split_type / f"train_{run_idx}.parquet"
                    ds_std = self.compute_ds_std(train_file)
                    if metric not in ["RMSE", "MAE"]:
                        ds_std = 1

                    vals.append(float(metrics_dict[metric] / ds_std))
            if vals:
                out[split_type] = self.aggregate(vals)
        return out


    def construct_full_stats_table(
        self,
        dir_path = None,
        metric = "RMSE", # or "MAE", None -> ds_std = 1
        baseline_only: bool = False,
        include_variants: bool = True,
    ):
        records = []
        results_root = self._resolve_results_root(dir_path)

        for model_label in self._list_model_labels(
            dir_path = results_root,
            baseline_only = baseline_only,
            include_variants = include_variants,
        ):
            model_name, variant_name = self._parse_model_label(model_label)
            data_loader = DataSaver(model_name, results_root)
            save_dir = self._resolve_variant_dir(model_name, variant_name, dir_path = results_root)

            for ds_name in self._list_dataset_names(save_dir):
                file_name = save_dir / f"{ds_name}.json"
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
            raise ValueError("No data loaded. Check ROOT, folder structure, and PRIMARY_METRIC.")

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


    # Hypothesis Testing Framework - Friedman's test
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


    def split_agnostic_test(self, long_df: pd.DataFrame = None):
        """
        This framework disregards the splitting strategies, exclude Random Split due to iid behaviour, for each dataset

        Null hypothesis: All models perform equally
        Alt hypothesis : At least 1 model performs significantly different
        """
        if long_df is None:
            long_df = self.construct_full_stats_table(metric = "RMSE", baseline_only = True)

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


    # -------------------------------
    # (LaTeX formatting)
    # -------------------------------
    def _fmt_rank_latex(self, val: float, tag: str) -> str:
        """tag in {'best','tie','normal'}."""
        if tag == "best":
            return rf"\textcolor{{red}}{{{val:.2f}}}"
        if tag == "tie":
            return rf"\textcolor{{blue}}{{{val:.2f}}}"
        return f"{val:.2f}"


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


    def split_wise_test(
        self,
        long_df: pd.DataFrame = None,
        baseline_only: bool = True,
        include_variants: bool = False,
    ):
        if long_df is None:
            long_df = self.construct_full_stats_table(
                metric = "RMSE",
                baseline_only = baseline_only,
                include_variants = include_variants,
            )

        summary_rows = []

        # buffer to build LaTeX rows at the end
        latex_buffer = {}  # {model: {split_type: (mean_rank, tag)}}

        split_types = sorted(long_df["split"].unique())

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
        print("\n" + "#" * 90)
        print("LATEX ROWS (mean ranks per split; red=best, blue=tied-with-best)\n")
        self.print_splitwise_meanrank_latex(latex_buffer, split_types)
