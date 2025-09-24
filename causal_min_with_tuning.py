#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal pipeline for estimating ATE and ITE of isoprene using DoWhy + EconML (CausalForestDML).
Includes simple hyperparameter tuning for LightGBM models.

Steps:
1. Load CSV with treatment, outcome, and covariates.
2. Define a user-specified DAG (edit function `user_dag_gml()`).
3. Tune LightGBM base learners for outcome (Y) and treatment (T).
4. Estimate ATE and ITE using CausalForestDML through DoWhy.
5. Save results (text summary and ITE CSV).

Author: <your name>
License: MIT
"""

import os
import argparse
from datetime import datetime
from typing import List, Dict

import pandas as pd
from dowhy import CausalModel
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV

RANDOM_STATE = 42


# ------------------------- User-defined DAG -------------------------
def user_dag_gml() -> str:
    """
    Define the causal DAG as a GML string.

    Important:
    - Replace `nodes_label` and `edges` with your own study design.
    - Labels must match your CSV column names for observed variables.
    - Node 'U' can represent unobserved confounding (not present in CSV).
    """
    nodes_label = ["RH", "Temp", "Radiation", "WS", "WD",
                   "Oxides", "Toluene", "Month", "U", "Isoprene"]
    nodes_alias = ["A", "B", "C", "D", "E", "F", "H", "I", "U", "Y"]

    # Example edges (alias notation). "AY" means A -> Y.
    edges = [
        "AY","BY","CY","DY","EY","FY","HY","IY",
        "AH","BA","BD","BE","BH","BF","CH","CF","DF","DH","HF",
        "UY"
    ]

    g = ["graph [directed 1"]
    for alias, label in zip(nodes_alias, nodes_label):
        g.append(f'  node [id "{alias}" label "{label}"]')
    for e in edges:
        g.append(f'  edge [source "{e[0]}" target "{e[1]}"]')
    g.append("]")
    return "\n".join(g)


# ------------------------- LightGBM tuning -------------------------
def tune_lgbm(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Tune LightGBM regressor using GridSearchCV and return the best parameters.
    """
    param_grid = {
        "max_depth": [3, 6, 10],
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.05, 0.1]
    }
    model = GridSearchCV(
        estimator=LGBMRegressor(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=10,
        n_jobs=-1,
        scoring="neg_mean_squared_error"
    )
    model.fit(X, y)
    return model.best_params_


# ------------------------- Main function -------------------------
def main():
    parser = argparse.ArgumentParser(description="Minimal causal pipeline with tuning.")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV path.")
    parser.add_argument("--treatment", type=str, required=True, help="Treatment variable.")
    parser.add_argument("--outcome", type=str, required=True, help="Outcome variable.")
    parser.add_argument("--outdir", type=str, default="./outputs", help="Output directory.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load dataset
    df = pd.read_csv(args.csv).reset_index(drop=True)

    # 2) Build DAG
    gml_string = user_dag_gml()

    # 3) Define covariates (all columns except treatment and outcome)
    covariates = [c for c in df.columns if c not in [args.treatment, args.outcome]]

    # 4) Hyperparameter tuning
    print("[Tuning] Outcome model (Y)...")
    best_y = tune_lgbm(df[covariates], df[args.outcome])
    print("Best params Y:", best_y)

    print("[Tuning] Treatment model (T)...")
    best_t = tune_lgbm(df[covariates], df[args.treatment])
    print("Best params T:", best_t)

    # 5) Build CausalModel
    model = CausalModel(
        data=df,
        treatment=args.treatment,
        outcome=args.outcome,
        graph=gml_string
    )
    estimand = model.identify_effect()

    estimate = model.estimate_effect(
        identified_estimand=estimand,
        method_name="backdoor.econml.dml.CausalForestDML",
        method_params={
            "init_params": {
                "model_y": LGBMRegressor(random_state=RANDOM_STATE, **best_y),
                "model_t": LGBMRegressor(random_state=RANDOM_STATE, **best_t),
                "cv": 4
            }
        }
    )

    ate_value = float(estimate.value)
    ite_series = pd.Series(estimate.cate_estimates, index=df.index, name="ITE")

    # 6) Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.outdir, f"causal_results_{ts}.txt"), "w") as f:
        f.write("===== Estimand =====\n")
        f.write(str(estimand) + "\n\n")
        f.write("===== ATE =====\n")
        f.write(f"{ate_value:.6f}\n\n")
        f.write("===== Best params =====\n")
        f.write("Y: " + str(best_y) + "\n")
        f.write("T: " + str(best_t) + "\n")

    ite_series.to_csv(os.path.join(args.outdir, f"ite_{ts}.csv"))

    print(f"ATE = {ate_value:.6f}")
    print("Results saved to", args.outdir)


if __name__ == "__main__":
    main()
