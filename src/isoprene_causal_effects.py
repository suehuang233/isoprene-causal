#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Causal effect estimation of environmental drivers on isoprene.

This script reproduces the DoWhy/EconML calculations used to estimate ATE and ITE
for two model settings:
1. with OH included;
2. without OH included.

"""

import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
from dowhy import CausalModel
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


# =============================================================================
# User configuration
# =============================================================================
SEED = 2026

# Place the input CSV in the same directory as this script, or replace this path.
DATA_PATH = "../data/month5to10-5to20hour-2020.csv"

OUTPUT_DIR = "../results/causal_effects"

MONTH_MIN, MONTH_MAX = 5, 10
HOUR_MIN, HOUR_MAX = 5, 20

OUTCOME = "Isoprene"

TREATMENTS_WITH_OH = [
    "RH", "Temp", "Radiation", "u", "v",
    "Oxides", "Toluene", "leaf_stage", "OH",
]

TREATMENTS_WITHOUT_OH = [
    "RH", "Temp", "Radiation", "u", "v",
    "Oxides", "Toluene", "leaf_stage",
]

DROP_COLS_WITH_OH = ["Year", "WS", "WD", "LAI", ]
DROP_COLS_WITHOUT_OH = ["Year", "WS", "WD", "LAI"]

# Leave as an empty list to run all treatments. Example: ONLY_RUN = ["Temp"]
ONLY_RUN = []

ITERATIONS = 3
REFUTE_THRESHOLD = 0.1


# =============================================================================
# Utilities
# =============================================================================
def set_all_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_text(path: str, text: str) -> None:
    """Save text output to file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def normalize_data_zscore(df: pd.DataFrame, exclude_columns=None) -> pd.DataFrame:
    """Apply z-score standardization to all columns except excluded columns."""
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = []

    cols = [c for c in df.columns if c not in exclude_columns]
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def tune_hyperparameters(
    df: pd.DataFrame,
    covariates: list,
    treatment_variable: str,
    outcome_variable: str,
):
    """Tune LightGBM models for the outcome and treatment equations."""
    param_grid = {
        "max_depth": [3, 6, 10],
        "n_estimators": [50, 150],
        "learning_rate": [0.05, 0.1],
        "min_child_samples": [20, 50],
    }

    model_y = GridSearchCV(
        estimator=LGBMRegressor(random_state=SEED),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    model_y.fit(df[covariates], df[outcome_variable])
    best_params_y = model_y.best_params_

    model_t = GridSearchCV(
        estimator=LGBMRegressor(random_state=SEED),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    model_t.fit(df[covariates], df[treatment_variable])
    best_params_t = model_t.best_params_

    return best_params_y, best_params_t


def get_ite_from_dowhy_estimate(
    estimate,
    df: pd.DataFrame,
    covariates: list,
) -> np.ndarray:
    """Extract ITE values from a DoWhy/EconML estimate object."""
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
            "Cannot find the underlying estimator inside the DoWhy estimate object."
        )

    econml_obj = None
    for attr in ["_econml_estimator", "econml_estimator", "_model", "model", "estimator"]:
        if hasattr(est, attr):
            econml_obj = getattr(est, attr)
            if econml_obj is not None:
                break

    if econml_obj is None:
        econml_obj = est

    x_values = df[covariates].values

    if hasattr(econml_obj, "const_marginal_effect"):
        ite = econml_obj.const_marginal_effect(x_values)
        return np.asarray(ite).reshape(-1)

    if hasattr(econml_obj, "effect"):
        t0 = np.zeros((len(df), 1))
        ite = econml_obj.effect(x_values, T0=t0, T1=t0 + 1e-3) / 1e-3
        return np.asarray(ite).reshape(-1)

    raise AttributeError(
        "The underlying estimator was found, but ITE could not be computed."
    )


# =============================================================================
# DAG specification
# =============================================================================
def build_global_gml(include_oh: bool = True):
    """Build the global DAG used by DoWhy."""
    if include_oh:
        nodes_label = [
            "RH", "Temp", "Radiation", "u", "v",
            "Oxides", "OH", "Toluene", "leaf_stage", "U", "Isoprene",
        ]
        nodes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "U", "Y"]
        edges = [
            "AY", "BY", "CY", "DY", "EY", "FY", "HY", "GY", "IY",
            "AF", "AG", "AH",
            "BA", "BH", "BF", "BG",
            "CH", "CF", "CG",
            "DF", "DG", "DH",
            "EF", "EG", "EH",
            "GF", "GH",
            "HF",
            "UY",
        ]
    else:
        nodes_label = [
            "RH", "Temp", "Radiation", "u", "v",
            "Oxides", "Toluene", "leaf_stage", "U", "Isoprene",
        ]
        nodes = ["A", "B", "C", "D", "E", "F", "H", "I", "U", "Y"]
        edges = [
            "AY", "BY", "CY", "DY", "EY", "FY", "HY", "IY",
            "AF", "AH",
            "BA", "BH", "BF",
            "CH", "CF",
            "DF", "DH",
            "EF", "EH",
            "HF",
            "UY",
        ]

    gml = "graph [directed 1\n"
    for node, label in zip(nodes, nodes_label):
        gml += f'\tnode [id "{node}" label "{label}"]\n'
    for edge in edges:
        gml += f'\tedge [source "{edge[0]}" target "{edge[1]}"]\n'
    gml += "]"

    return gml, nodes_label


# =============================================================================
# Causal estimation
# =============================================================================
def estimate_causal_ite_dowhy(
    df: pd.DataFrame,
    treatment_variable: str,
    outcome_variable: str,
    gml_string: str,
    nodes_label: list,
    best_params_y: dict,
    best_params_t: dict,
    threshold: float = 0.1,
):
    """Estimate ATE and ITE using DoWhy with EconML CausalForestDML."""
    covariates = nodes_label.copy()
    if "U" in covariates:
        covariates.remove("U")

    for var in [treatment_variable, outcome_variable]:
        if var in covariates:
            covariates.remove(var)

    df = df.copy()
    df[outcome_variable] = df[outcome_variable].values.ravel()

    model = CausalModel(
        data=df,
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
                "model_y": LGBMRegressor(**best_params_y, random_state=SEED),
                "model_t": LGBMRegressor(**best_params_t, random_state=SEED),
                "cv": 4,
                "random_state": SEED,
            },
            "fit_params": {},
        },
    )

    ate_value = float(estimate.value)
    ite_values = get_ite_from_dowhy_estimate(estimate, df, covariates)

    ref_subset = model.refute_estimate(
        estimand,
        estimate,
        method_name="data_subset_refuter",
        method_params={"random_seed": SEED},
    )

    ref_rcc = model.refute_estimate(
        estimand,
        estimate,
        method_name="random_common_cause",
        method_params={"random_seed": SEED},
    )

    dowhy_controls = estimand.get_backdoor_variables()

    warn = False
    for ref in [ref_subset, ref_rcc]:
        if hasattr(ref, "effect_difference") and abs(ref.effect_difference) > threshold:
            warn = True
        if hasattr(ref, "refutation_score") and abs(ref.refutation_score) > threshold:
            warn = True

    return ate_value, ite_values, ref_subset, ref_rcc, estimand, dowhy_controls, warn


def save_results_to_file(
    estimand,
    ref_subset,
    ref_rcc,
    best_params_y: dict,
    best_params_t: dict,
    file_path: str,
) -> str:
    """Save causal estimand, refutation results, and model parameters."""
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


# =============================================================================
# Analysis runner
# =============================================================================
def run_analysis(
    data_path: str,
    output_dir: str,
    treatments: list,
    drop_cols: list,
    include_oh: bool,
) -> None:
    """Run the full ATE/ITE workflow for one DAG setting."""
    ensure_dir(output_dir)
    gml_string, nodes_label_model = build_global_gml(include_oh=include_oh)

    df = pd.read_csv(data_path)

    df = df[(df["Month"] >= MONTH_MIN) & (df["Month"] <= MONTH_MAX)]
    df = df[(df["Hour"] >= HOUR_MIN) & (df["Hour"] <= HOUR_MAX)]
    df = df.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)

    needed = ["Month", "Hour", OUTCOME] + treatments
    missing_cols = [col for col in needed if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input data: {missing_cols}")

    df = df[needed].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[OUTCOME] + treatments).reset_index(drop=True)

    df_norm = normalize_data_zscore(df, exclude_columns=["Month", "Hour"])

    ate_rows = []
    controls_used = []
    run_list = treatments if not ONLY_RUN else ONLY_RUN

    for treat in run_list:
        if treat not in df_norm.columns:
            print(f"[SKIP] {treat} is not in the dataframe columns.")
            continue

        print("\n" + "=" * 80)
        print(f"Treatment: {treat}")

        covariates_for_tune = nodes_label_model.copy()
        covariates_for_tune.remove("U")
        for var in [treat, OUTCOME]:
            if var in covariates_for_tune:
                covariates_for_tune.remove(var)

        best_params_y, best_params_t = tune_hyperparameters(
            df_norm,
            covariates_for_tune,
            treat,
            OUTCOME,
        )

        ate_list = []
        ite_df = pd.DataFrame({treat: df[treat].values})
        warn_any = False

        for i in range(ITERATIONS):
            set_all_seeds(SEED + i)

            ate, ite, ref_subset, ref_rcc, estimand, cov_used, warn = estimate_causal_ite_dowhy(
                df_norm,
                treat,
                OUTCOME,
                gml_string,
                nodes_label_model,
                best_params_y,
                best_params_t,
                threshold=REFUTE_THRESHOLD,
            )

            warn_any = warn_any or warn
            ate_list.append(ate)
            ite_df[f"ITE_{treat}_{i}"] = ite

            save_results_to_file(
                estimand,
                ref_subset,
                ref_rcc,
                best_params_y,
                best_params_t,
                file_path=os.path.join(output_dir, f"{treat}_iter{i}_refutation.txt"),
            )

            subset_diff = getattr(ref_subset, "effect_difference", None)
            rcc_diff = getattr(ref_rcc, "effect_difference", None)
            print(
                f"[iter {i}] ATE={ate:.4f} | "
                f"subset_diff={subset_diff} | rcc_diff={rcc_diff} | warn={warn}"
            )

        ate_mean = float(np.mean(ate_list))
        ate_sd = float(np.std(ate_list, ddof=1)) if len(ate_list) > 1 else 0.0
        ate_rows.append(
            {
                "treatment": treat,
                "ate_mean": ate_mean,
                "ate_std": ate_sd,
            }
        )

        controls_used.append(
            {
                "treatment": treat,
                "controls": ", ".join(cov_used),
                "warn": warn_any,
            }
        )

        ite_out = os.path.join(output_dir, f"ITE_{treat}_iters{ITERATIONS}.csv")
        ite_df.to_csv(ite_out, index=False)
        print("Saved ITE:", ite_out)

    ate_summary = pd.DataFrame(ate_rows)
    ate_summary_path = os.path.join(output_dir, f"ATE_summary_iters{ITERATIONS}.csv")
    ate_summary.to_csv(ate_summary_path, index=False)

    controls_df = pd.DataFrame(controls_used)
    controls_path = os.path.join(output_dir, "controls_used.csv")
    controls_df.to_csv(controls_path, index=False)

    print("\nDone. Outputs saved to:", output_dir)
    print("ATE summary:", ate_summary_path)
    print("Controls used:", controls_path)


def main() -> None:
    """Run both original analysis settings."""
    set_all_seeds(SEED)

    run_analysis(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        treatments=TREATMENTS_WITH_OH,
        drop_cols=DROP_COLS_WITH_OH,
        include_oh=True,
    )

    run_analysis(
        data_path=DATA_PATH,
        output_dir=os.path.join(OUTPUT_DIR, "without_OH"),
        treatments=TREATMENTS_WITHOUT_OH,
        drop_cols=DROP_COLS_WITHOUT_OH,
        include_oh=False,
    )


if __name__ == "__main__":
    main()
