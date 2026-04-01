import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from collections import defaultdict
import json


class DataSaver:
    def __init__(self, model_name):
        self.model_name = model_name
        self.output_dir = "Results/"
        os.makedirs(self.output_dir, exist_ok = True)


    def _to_python(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if hasattr(obj, "item"):   # torch / numpy scalar
            return obj.item()
        raise TypeError


    def save_result(self, out_file, results):
        # Save path:  Results/{model_name}/{file_name}.json
        # where in each .json file, it encompasses of results for all types of split
        try:
            with open(out_file, "w") as f:
                json.dump(results, f, indent = 2, default = self._to_python)
            # tqdm.write(f"Successfully saved → {out_file}")
        except:
            tqdm.write(f"Error: Cannot save file")


    def read_json(self, file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
        return defaultdict(dict, data)


class AnalystModel:
    def __init__(
        self,
        alpha: float = 0.05,
        agg_method: str = "median",   # or mean (less robust)
    ):
        self.alpha = alpha            # significance level
        self.agg_method = agg_method


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


    def build_wide_numeric(self, metric: str, data_loader: DataSaver, save_dir: str, model_name: str) -> pd.DataFrame:
        records = []
        for ds_name in tqdm(DATASET_LIST, desc=f"Processing {model_name} ({metric})"):
            file_name = os.path.join(save_dir, f"{ds_name}.json")
            if not os.path.exists(file_name):
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
            raise ValueError(f"No data loaded for model={model_name}, metric={metric}.")

        wide = long_df.pivot_table(
            index="dataset",
            columns="split",
            values="score",
            aggfunc="first"
        ).reindex(index=DATASET_LIST, columns=SPLIT_TYPES)

        return wide


    def performance_table(self, model_name):
        if model_name not in MODEL_REGISTRY.keys():
            raise ValueError(f"model_name not exist. Must be {MODEL_REGISTRY.keys()}")

        save_dir = os.path.join("Results/", model_name)
        data_loader = DataSaver(model_name)

        if not os.path.exists(save_dir):
            raise FileExistsError(f"The file path {save_dir} does not exist")

        # ---------- numeric tables ----------
        rmse_num = self.build_wide_numeric("RMSE", data_loader, save_dir, model_name)   # nRMSE
        mae_num  = self.build_wide_numeric("MAE", data_loader, save_dir, model_name)    # nMAE
        smape_num = self.build_wide_numeric("sMAPE", data_loader, save_dir, model_name) # sMAPE (scaled)

        # ---------- RMSE-based minima for coloring ----------
        rmse_row_min = rmse_num.min(axis=1, skipna=True)  # best split per dataset
        rmse_col_min = rmse_num.min(axis=0, skipna=True)  # best dataset per split

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

                # RMSE-only coloring decision
                is_row_best = (r_txt != "--") and (r == rmse_row_min.loc[ds])
                is_col_best = (r_txt != "--") and (r == rmse_col_min.loc[split])

                cell_colored = self.colorize_cell(cell_plain, is_row_best, is_col_best)

                row_cells.append(f"${cell_colored}$")

            latex_rows.append(" & ".join(row_cells) + " \\\\")
        print("\n\n".join(latex_rows))


    def side_exp_performance_table(self, model_name):
        if model_name not in MODEL_REGISTRY.keys():
            raise ValueError(f"model_name not exist. Must be {MODEL_REGISTRY.keys()}")

        save_dir = os.path.join("Results/", model_name)
        data_loader = DataSaver(model_name)

        if not os.path.exists(save_dir):
            raise FileExistsError(f"The file path {save_dir} does not exist")

        # ---------- baseline numeric tables ----------
        save_dir = os.path.join("Results/", model_name)
        base_rmse_num = self.build_wide_numeric("RMSE", data_loader, save_dir, model_name)   # nRMSE

        # ---------- side experiment numeric tables ----------
        save_dir = os.path.join("Results_add/", model_name)
        rmse_num = self.build_wide_numeric("RMSE", data_loader, save_dir, model_name)   # nRMSE
        mae_num  = self.build_wide_numeric("MAE", data_loader, save_dir, model_name)    # nMAE
        smape_num = self.build_wide_numeric("sMAPE", data_loader, save_dir, model_name) # sMAPE (scaled)

        # ---------- generate LaTeX rows: each cell = nRMSE / nMAE / sMAPE, colored ONLY by RMSE ----------
        latex_rows = []
        for i, ds in enumerate(rmse_num.index):
            row_cells = [f"\\#{i+1}\n"]

            for split in rmse_num.columns:
                r = rmse_num.loc[ds, split]
                base_r = base_rmse_num.loc[ds, split]
                m = mae_num.loc[ds, split]
                s = smape_num.loc[ds, split]

                # If any metric missing, show "-- / -- / --" (still color by RMSE if RMSE exists)
                r_txt = self.adaptive_format(r)
                m_txt = self.adaptive_format(m)
                s_txt = self.adaptive_format(s)

                diff_percent = int((r - base_r) / base_r * 100)
                cell_plain = f"{r_txt} / {m_txt} / {s_txt} ({diff_percent}\%)"

                # RMSE-only coloring decision
                cell_colored = f"\\textcolor{{ForestGreen}}{{{cell_plain}}}" if diff_percent < 0 else f"\\textcolor{{BrickRed}}{{{cell_plain}}}"
                if diff_percent == 0:
                    cell_colored = f"\\textcolor{{black}}{{{cell_plain}}}"

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
            # Comment out this to evaluate all type of splitting regimes
            if split_type not in SPLIT_TYPES:
                continue
            vals = []
            for run_idx, metrics_dict in runs.items():
                if metric in metrics_dict and metrics_dict[metric] is not None:
                    train_file = os.path.join("../data/splitted", ds_name, split_type, f"train_{run_idx}.parquet")
                    ds_std = self.compute_ds_std(train_file)
                    if metric not in ["RMSE", "MAE"]:
                        ds_std = 1

                    vals.append(float(metrics_dict[metric] / ds_std))
            if vals:
                out[split_type] = self.aggregate(vals)
        return out


    def construct_full_stats_table(
        self,
        dir_path = "Results/",
        metric = "RMSE" # or "MAE", None -> ds_std = 1
    ):
        records = []

        for model_name in tqdm(MODEL_REGISTRY.keys(), desc = f"Processing: "):
            save_dir = os.path.join(dir_path, model_name)
            data_loader = DataSaver(model_name)

            if not os.path.exists(save_dir):
                raise FileExistsError(f"The file path {save_dir} does not exist")

            for ds_name in tqdm(DATASET_LIST, desc = f"Processing {model_name}"):
                file_name = os.path.join(save_dir, f"{ds_name}.json")

                res_dict = data_loader.read_json(file_name)
                split_scores = self.split_score_by_dict(res_dict, metric, ds_name)

                for split_type, score in split_scores.items():
                    records.append({
                        "dataset": ds_name,
                        "split": split_type,
                        "model": model_name,
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
        if split_type not in SPLIT_TYPES:
            raise ValueError(f"{split_type} does not exist. The values must be {SPLIT_TYPES}")

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
            long_df = self.construct_full_stats_table(metric = "RMSE")

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
        for model in MODEL_REGISTRY.keys():
            if model not in latex_buffer:
                raise ValueError("model does not exist")
            # Escape underscores for LaTeX
            model_name = str(model).replace("_", r"\_")
            row = [model_name]
            for s in split_types:
                rank, tag = latex_buffer[model][s]
                row.append(self._fmt_rank_latex(float(rank), tag))
            print(" & ".join(row) + r" \\")
            print()  # blank line between models


    def split_wise_test(self, long_df: pd.DataFrame = None):
        if long_df is None:
            long_df = self.construct_full_stats_table(metric="RMSE")

        summary_rows = []

        # buffer to build LaTeX rows at the end
        latex_buffer = {}  # {model: {split_type: (mean_rank, tag)}}

        for split_type in SPLIT_TYPES:
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
        self.print_splitwise_meanrank_latex(latex_buffer, SPLIT_TYPES)