#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal ATE/ITE estimation for isoprene using DoWhy + EconML (CausalForestDML).
Includes LightGBM hyperparameter tuning and a basic refutation (data_subset_refuter).

Steps:
1) Load CSV with treatment, outcome, and covariates.
2) Define a user-specified DAG (edit function `user_dag_gml()`).
3) Tune LightGBM base learners for outcome (Y) and treatment (T).
4) Estimate ATE and ITE using CausalForestDML through DoWhy.
5) Run a simple refutation test (data_subset_refuter).
6) Save results (text summary with refutation + ITE CSV) and print absolute paths.

Author: <Shu HUANG>
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
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt

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
        "max_depth": [3, 10, 20, 100],
        "n_estimators": [10, 50, 100],
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
    parser = argparse.ArgumentParser(description="Minimal causal pipeline with tuning and refutation.")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV path.")
    parser.add_argument("--treatment", type=str, required=True, help="Treatment variable.")
    parser.add_argument("--outcome", type=str, required=True, help="Outcome variable.")
    parser.add_argument("--outdir", type=str, default="./outputs", help="Output directory.")
    parser.add_argument("--standardize", action="store_true", help="Standardize the dataset (use StandardScaler).")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load dataset
    df = pd.read_csv(args.csv).reset_index(drop=True)
    
    # 2) Standardize data if needed
    if args.standardize:
        print("[Standardizing] Using StandardScaler ...")
        scaler = StandardScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])

    # 3) Build DAG
    gml_string = user_dag_gml()

    # 4) Define covariates (all columns except treatment and outcome)
    covariates = [c for c in df.columns if c not in [args.treatment, args.outcome]]

    # 5) Hyperparameter tuning
    print("[Tuning] Outcome model (Y)...")
    best_y = tune_lgbm(df[covariates], df[args.outcome])
    print("Best params Y:", best_y)

    print("[Tuning] Treatment model (T)...")
    best_t = tune_lgbm(df[covariates], df[args.treatment])
    print("Best params T:", best_t)

    # 6) Build CausalModel and estimate ATE/ITE
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
            },
            # Important: DoWhy's econml adapter expects this key to exist.
            "fit_params": {}
        }
    )

    ate_value = float(estimate.value)
    ite_series = pd.Series(estimate.cate_estimates, index=df.index, name="ITE")
    print(f"ATE = {ate_value:.6f}")

    # 7) Refutation: data_subset_refuter (simple robustness check)
    print("[Refutation] Running data_subset_refuter ...")
    refutation = model.refute_estimate(
        estimand,
        estimate,
        method_name="data_subset_refuter"
    )
    print(refutation)

    # 8) Save results and print absolute paths
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(args.outdir, f"causal_results_{ts}.txt")
    ite_path = os.path.join(args.outdir, f"ite_{ts}.csv")

    with open(txt_path, "w") as f:
        f.write("===== Estimand =====\n")
        f.write(str(estimand) + "\n\n")
        f.write("===== ATE =====\n")
        f.write(f"{ate_value:.6f}\n\n")
        f.write("===== Best params =====\n")
        f.write("Y: " + str(best_y) + "\n")
        f.write("T: " + str(best_t) + "\n\n")
        f.write("===== Refutation Results (data_subset_refuter) =====\n")
        f.write(str(refutation) + "\n")

    ite_series.to_csv(ite_path)

    print("[Saved]", os.path.abspath(txt_path))
    print("[Saved]", os.path.abspath(ite_path))
    print("Results saved to", os.path.abspath(args.outdir))
    
    # 9) Save the causal graph as a PNG image
    graph_path = os.path.join(args.outdir, f"causal_graph_{ts}.png")
    print(f"[Saving] Causal graph to {graph_path}")

    # Draw the causal graph and save it to PNG
    G = nx.parse_gml(gml_string)
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=12, font_weight='bold', edge_color='gray')
    plt.title('Causal DAG', fontsize=15)
    plt.savefig(graph_path, format='PNG')
    plt.close()

    print("Results saved to", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
