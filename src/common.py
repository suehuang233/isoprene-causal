import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from dowhy import CausalModel
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class AnalysisConfig:
    seed: int
    data_path: str
    output_dir: str
    month_min: int
    month_max: int
    hour_min: int
    hour_max: int
    outcome: str
    treatments: List[str]
    drop_cols: List[str]
    only_run: List[str]
    iterations: int
    refute_threshold: float
    plot_dag: bool
    nodes_label: List[str]
    nodes: List[str]
    edges: List[str]
    dag_png_name: str


# =========================
# Utilities
# =========================
def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def normalize_data_zscore(df: pd.DataFrame, exclude_columns=None) -> pd.DataFrame:
    """Z-score standardize columns except exclude_columns."""
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = []
    cols = [c for c in df.columns if c not in exclude_columns]
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def tune_hyperparameters(
    df: pd.DataFrame,
    covariates: List[str],
    treatment_variable: str,
    outcome_variable: str,
    seed: int,
) -> Tuple[dict, dict]:
    """
    Minimal-change version: keep the original style and parameter search space.
    """
    param_grid = {
        "max_depth": [3, 6, 10],
        "n_estimators": [50, 150],
        "learning_rate": [0.05, 0.1],
        "min_child_samples": [20, 50],
    }

    model_y = GridSearchCV(
        estimator=LGBMRegressor(random_state=seed),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    model_y.fit(df[covariates], df[outcome_variable])
    best_params_y = model_y.best_params_

    model_t = GridSearchCV(
        estimator=LGBMRegressor(random_state=seed),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    model_t.fit(df[covariates], df[treatment_variable])
    best_params_t = model_t.best_params_

    return best_params_y, best_params_t


def get_ite_from_dowhy_estimate(estimate, df: pd.DataFrame, covariates: List[str]) -> np.ndarray:
    if hasattr(estimate, "cate_estimates") and estimate.cate_estimates is not None:
        return np.asarray(estimate.cate_estimates).reshape(-1)

    est = None
    for attr in ["estimator", "_estimator_object", "_estimator", "causal_estimator"]:
        if hasattr(estimate, attr):
            est = getattr(estimate, attr)
            if est is not None:
                break

    if est is None:
        raise AttributeError(
            "Cannot find underlying estimator inside DoWhy estimate. "
            "Please print(dir(estimate)) to inspect available fields."
        )

    econml_obj = None
    for attr in ["_econml_estimator", "econml_estimator", "_model", "model", "estimator"]:
        if hasattr(est, attr):
            econml_obj = getattr(est, attr)
            if econml_obj is not None:
                break

    if econml_obj is None:
        econml_obj = est

    X = df[covariates].values

    if hasattr(econml_obj, "const_marginal_effect"):
        ite = econml_obj.const_marginal_effect(X)
        return np.asarray(ite).reshape(-1)

    if hasattr(econml_obj, "effect"):
        T0 = np.zeros((len(df), 1))
        ite = econml_obj.effect(X, T0=T0, T1=T0 + 1e-3) / 1e-3
        return np.asarray(ite).reshape(-1)

    raise AttributeError(
        "Underlying estimator found, but cannot compute ITE (no const_marginal_effect/effect)."
    )


def build_global_gml(nodes: List[str], nodes_label: List[str], edges: List[str]):
    gml = 'graph [directed 1\n'
    for node, label in zip(nodes, nodes_label):
        gml += f'\tnode [id "{node}" label "{label}"]\n'
    for e in edges:
        gml += f'\tedge [source "{e[0]}" target "{e[1]}"]\n'
    gml += ']'
    return gml, nodes, nodes_label, edges


def plot_global_dag(gml_string: str, out_png: str) -> None:
    G = nx.parse_gml(gml_string)
    pos = nx.spring_layout(G, seed=7)

    labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=2200,
        node_color="white",
        edgecolors="black",
        linewidths=1.2,
    )
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=11)
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=18,
        width=1.2,
        edge_color="black",
    )
    plt.axis("off")
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=600)
    plt.close()


def estimate_causal_ite_dowhy(
    df1: pd.DataFrame,
    treatment_variable: str,
    outcome_variable: str,
    gml_string: str,
    nodes_label: List[str],
    best_params_y: dict,
    best_params_t: dict,
    threshold: float,
    seed: int,
):
    nodes_label_0 = nodes_label.copy()
    if "U" in nodes_label_0:
        nodes_label_0.remove("U")

    for var in [treatment_variable, outcome_variable]:
        if var in nodes_label_0:
            nodes_label_0.remove(var)

    covariates = nodes_label_0

    df1 = df1.copy()
    df1[outcome_variable] = df1[outcome_variable].values.ravel()

    model = CausalModel(
        data=df1,
        treatment=treatment_variable,
        outcome=outcome_variable,
        graph=gml_string,
    )

    estimand = model.identify_effect(proceed_when_unidentifiable=True)

    estimate = model.estimate_effect(
        identified_estimand=estimand,
        method_name="backdoor.econml.dml.CausalForestDML",
        method_params={
            "init_params": {
                "model_y": LGBMRegressor(**best_params_y, random_state=seed),
                "model_t": LGBMRegressor(**best_params_t, random_state=seed),
                "cv": 4,
                "random_state": seed,
            },
            "fit_params": {},
        },
    )
    print("DoWhy estimate type:", type(estimate))
    print("DoWhy estimate dir has estimator?", hasattr(estimate, "estimator"))
    print("Keys:", [k for k in dir(estimate) if "est" in k.lower()])

    ate_value = float(estimate.value)
    ite_values = get_ite_from_dowhy_estimate(estimate, df1, covariates)

    ref_subset = model.refute_estimate(
        estimand,
        estimate,
        method_name="data_subset_refuter",
        method_params={"random_seed": seed},
    )

    ref_rcc = model.refute_estimate(
        estimand,
        estimate,
        method_name="random_common_cause",
        method_params={"random_seed": seed},
    )

    dowhy_w = estimand.get_backdoor_variables()

    warn = False
    for ref in [ref_subset, ref_rcc]:
        if hasattr(ref, "effect_difference") and abs(ref.effect_difference) > threshold:
            warn = True
        if hasattr(ref, "refutation_score") and abs(ref.refutation_score) > threshold:
            warn = True

    return ate_value, ite_values, ref_subset, ref_rcc, estimand, dowhy_w, warn


def save_results_to_file(
    estimand,
    ref_subset,
    ref_rcc,
    best_params_y: dict,
    best_params_t: dict,
    file_path: str,
) -> str:
    ensure_dir(os.path.dirname(file_path))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.basename(file_path)
    name, ext = os.path.splitext(base)
    out = os.path.join(os.path.dirname(file_path), f"{name}_{timestamp}{ext}")

    txt = []
    txt.append("===== Causal Estimand =====\n")
    txt.append(str(estimand) + "\n\n")
    txt.append("===== Refutation Results: data_subset_refuter =====\n")
    txt.append(str(ref_subset) + "\n\n")
    txt.append("===== Refutation Results: random_common_cause_refuter =====\n")
    txt.append(str(ref_rcc) + "\n\n")
    txt.append("===== Best Params =====\n")
    txt.append("best_params_t:\n" + str(best_params_t) + "\n\n")
    txt.append("best_params_y:\n" + str(best_params_y) + "\n\n")
    save_text(out, "".join(txt))
    return out


def run_analysis(config: AnalysisConfig) -> None:
    set_all_seeds(config.seed)
    ensure_dir(config.output_dir)

    gml_string, _, _, _ = build_global_gml(config.nodes, config.nodes_label, config.edges)
    if config.plot_dag:
        plot_global_dag(gml_string, os.path.join(config.output_dir, config.dag_png_name))

    df = pd.read_csv(config.data_path)

    df = df[(df["Month"] >= config.month_min) & (df["Month"] <= config.month_max)]
    df = df[(df["Hour"] >= config.hour_min) & (df["Hour"] <= config.hour_max)]

    df = df.drop(columns=config.drop_cols, errors="ignore").reset_index(drop=True)

    needed = ["Month", "Hour", config.outcome] + config.treatments
    needed = [c for c in needed if c in df.columns]
    df = df[needed].copy()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=[config.outcome] + [t for t in config.treatments if t in df.columns]
    ).reset_index(drop=True)

    exclude = ["Month", "Hour"]
    df_norm = normalize_data_zscore(df, exclude_columns=exclude)

    controls_used = []
    ate_rows = []
    run_list = config.treatments if (not config.only_run) else config.only_run

    for treat in run_list:
        if treat not in df_norm.columns:
            print(f"[SKIP] {treat} not in dataframe columns.")
            continue

        print("\n" + "=" * 80)
        print(f"Treatment: {treat}")

        nodes_label_model = config.nodes_label.copy()
        nodes_label_0 = nodes_label_model.copy()
        nodes_label_0.remove("U")
        for var in [treat, config.outcome]:
            if var in nodes_label_0:
                nodes_label_0.remove(var)
        covariates_for_tune = nodes_label_0

        best_params_y, best_params_t = tune_hyperparameters(
            df_norm,
            covariates_for_tune,
            treat,
            config.outcome,
            config.seed,
        )

        ate_list = []
        ite_df = pd.DataFrame({treat: df[treat].values})

        warn_any = False
        for i in range(config.iterations):
            set_all_seeds(config.seed + i)
            ate, ite, ref_subset, ref_rcc, estimand, cov_used, warn = estimate_causal_ite_dowhy(
                df_norm,
                treat,
                config.outcome,
                gml_string,
                nodes_label_model,
                best_params_y,
                best_params_t,
                config.refute_threshold,
                config.seed,
            )
            warn_any = warn_any or warn
            ate_list.append(ate)
            ite_df[f"ITE_{i+1}"] = ite

            save_results_to_file(
                estimand,
                ref_subset,
                ref_rcc,
                best_params_y,
                best_params_t,
                file_path=os.path.join(config.output_dir, f"{treat}_refutation.txt"),
            )

            subset_diff = getattr(ref_subset, "effect_difference", None)
            rcc_diff = getattr(ref_rcc, "effect_difference", None)
            print(
                f"[iter {i}] ATE={ate:.4f} | subset_diff={subset_diff} | "
                f"rcc_diff={rcc_diff} | warn={warn}"
            )
        
        ate_mean = float(np.mean(ate_list))
        ate_sd = float(np.std(ate_list, ddof=1)) if len(ate_list) > 1 else 0.0
        ate_rows.append({"treatment": treat, "ate_mean": ate_mean, "ate_std": ate_sd})

        controls_used.append({"treatment": treat, "controls": ", ".join(cov_used), "warn": warn_any})

        ite_cols = [c for c in ite_df.columns if c.startswith("ITE_")]
        ite_df["ITE_mean"] = ite_df[ite_cols].mean(axis = 1)
        
        ite_out = os.path.join(config.output_dir, f"{treat}_ITE.csv")
        ite_df.to_csv(ite_out, index=False)
        print("Saved ITE:", ite_out)

    ate_summary = pd.DataFrame(ate_rows)
    ate_summary_path = os.path.join(config.output_dir, f"ATE_summary_iters{config.iterations}.csv")
    ate_summary.to_csv(ate_summary_path, index=False)

    controls_df = pd.DataFrame(controls_used)
    controls_path = os.path.join(config.output_dir, "controls_used.csv")
    controls_df.to_csv(controls_path, index=False)

    print("\nDone. Outputs saved to:", config.output_dir)
    print("ATE summary:", ate_summary_path)
    print("Controls used:", controls_path)
