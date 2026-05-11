#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sensitivity analysis for the causal effect estimates of environmental drivers on isoprene.

This script runs DoWhy's add_unobserved_common_cause refuter using direct simulation.
The preprocessing and DAG specification follow the main DoWhy/EconML analysis, except
that the explicit unobserved node U and the edge U -> Isoprene are removed for the
sensitivity-analysis refuter.
"""

import os
import random

import numpy as np
import pandas as pd
from dowhy import CausalModel
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler


# =============================================================================
# User configuration
# =============================================================================
SEED = 2026

# Place the input CSV in the same directory as this script, or replace this path.
DATA_PATH = "../data/month5to10-5to20hour-2020.csv"

OUTPUT_DIR = "../results/sensitivity_analysis"

MONTH_MIN, MONTH_MAX = 5, 10
HOUR_MIN, HOUR_MAX = 5, 20

OUTCOME = "Isoprene"
TREATMENTS = [
    "RH", "Temp", "Radiation", "u", "v",
    "Oxides", "Toluene", "leaf_stage", "OH",
]

DROP_COLS = ["Year", "WS", "WD", "LAI"]

# Sensitivity-grid settings. These coefficients are interpreted in standardized units
# because the analysis variables are z-score standardized below.
MAX_STRENGTH_1D = 1.0
N_POINTS_1D = 41

MAX_STRENGTH_2D = 1.0
N_POINTS_2D = 41

SAVE_NUMERIC_GRID = True

CF_INIT_PARAMS = dict(
    model_y=LGBMRegressor(random_state=SEED),
    model_t=LGBMRegressor(random_state=SEED),
    n_estimators=500,
    min_var_fraction_leaf=0.1,
    random_state=SEED,
)


# =============================================================================
# Utilities
# =============================================================================
def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def set_all_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


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


# =============================================================================
# DAG specification
# =============================================================================
def build_gml_no_u() -> str:
    """Build the DAG used for the sensitivity analysis, excluding the explicit U node."""
    nodes_label = [
        "RH", "Temp", "Radiation", "u", "v",
        "Oxides", "OH", "Toluene", "leaf_stage", "Isoprene",
    ]
    nodes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "Y"]

    edges = [
        "AY", "BY", "CY", "DY", "EY", "FY", "HY", "GY", "IY",
        "AF", "AG", "AH",
        "BA", "BH", "BF", "BG",
        "CH", "CF", "CG",
        "DF", "DG", "DH",
        "EF", "EG", "EH",
        "GF", "GH",
        "HF",
    ]

    gml = "graph [directed 1\n"
    for node, label in zip(nodes, nodes_label):
        gml += f'\tnode [id "{node}" label "{label}"]\n'
    for edge in edges:
        gml += f'\tedge [source "{edge[0]}" target "{edge[1]}"]\n'
    gml += "]"

    return gml


# =============================================================================
# Data preprocessing
# =============================================================================
def preprocess_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess data using the same filtering and standardization as the main analysis."""
    df = pd.read_csv(DATA_PATH)

    df = df[(df["Month"] >= MONTH_MIN) & (df["Month"] <= MONTH_MAX)]
    df = df[(df["Hour"] >= HOUR_MIN) & (df["Hour"] <= HOUR_MAX)]
    df = df.drop(columns=DROP_COLS, errors="ignore").reset_index(drop=True)

    needed = ["Month", "Hour", OUTCOME] + TREATMENTS
    missing_cols = [col for col in needed if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input data: {missing_cols}")

    df = df[needed].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[OUTCOME] + TREATMENTS).reset_index(drop=True)

    df_norm = normalize_data_zscore(df, exclude_columns=["Month", "Hour"])
    return df, df_norm


# =============================================================================
# Sensitivity-analysis calculations
# =============================================================================
def run_line_grid(
    model: CausalModel,
    estimand,
    estimate,
    treatment: str,
    base_ate: float,
    strengths: np.ndarray,
) -> pd.DataFrame:
    """Compute one-dimensional sensitivity results without generating plots."""
    rows = []

    for strength_t in strengths:
        ref = model.refute_estimate(
            estimand,
            estimate,
            method_name="add_unobserved_common_cause",
            simulation_method="direct-simulation",
            confounders_effect_on_treatment="linear",
            confounders_effect_on_outcome="linear",
            effect_strength_on_treatment=float(strength_t),
            effect_strength_on_outcome=0.0,
            random_seed=SEED,
            plotmethod=None,
        )

        new_ate = float(ref.new_effect)
        effect_difference = (
            float(ref.effect_difference)
            if hasattr(ref, "effect_difference")
            else new_ate - base_ate
        )

        rows.append(
            {
                "treatment": treatment,
                "base_ate": float(base_ate),
                "strength_t": float(strength_t),
                "strength_y": 0.0,
                "new_ate": new_ate,
                "effect_difference": effect_difference,
            }
        )

    return pd.DataFrame(rows)


def run_numeric_grid(
    model: CausalModel,
    estimand,
    estimate,
    treatment: str,
    base_ate: float,
    strengths: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    """Compute a two-dimensional sensitivity grid and identify the grid-based tipping point."""
    rows = []

    for strength_t in strengths:
        for strength_y in strengths:
            ref = model.refute_estimate(
                estimand,
                estimate,
                method_name="add_unobserved_common_cause",
                simulation_method="direct-simulation",
                confounders_effect_on_treatment="linear",
                confounders_effect_on_outcome="linear",
                effect_strength_on_treatment=float(strength_t),
                effect_strength_on_outcome=float(strength_y),
                random_seed=SEED,
                plotmethod=None,
            )

            new_ate = float(ref.new_effect)
            effect_difference = (
                float(ref.effect_difference)
                if hasattr(ref, "effect_difference")
                else new_ate - base_ate
            )

            rows.append(
                {
                    "treatment": treatment,
                    "base_ate": float(base_ate),
                    "strength_t": float(strength_t),
                    "strength_y": float(strength_y),
                    "new_ate": new_ate,
                    "effect_difference": effect_difference,
                }
            )

    grid = pd.DataFrame(rows)
    tip = find_tipping_point(grid, treatment, base_ate)
    return grid, tip


def find_tipping_point(grid: pd.DataFrame, treatment: str, base_ate: float) -> dict:
    """Find the smallest grid-point norm where the estimated effect crosses zero."""
    if base_ate > 0:
        candidates = grid[grid["new_ate"] <= 0].copy()
    else:
        candidates = grid[grid["new_ate"] >= 0].copy()

    if len(candidates) == 0:
        return {
            "treatment": treatment,
            "base_ate": float(base_ate),
            "tipping_strength_t": None,
            "tipping_strength_y": None,
            "tipping_new_ate": None,
        }

    candidates["norm"] = np.sqrt(
        candidates["strength_t"] ** 2 + candidates["strength_y"] ** 2
    )
    best = candidates.sort_values("norm").iloc[0]

    return {
        "treatment": treatment,
        "base_ate": float(base_ate),
        "tipping_strength_t": float(best["strength_t"]),
        "tipping_strength_y": float(best["strength_y"]),
        "tipping_new_ate": float(best["new_ate"]),
    }


# =============================================================================
# Main runner
# =============================================================================
def main() -> None:
    """Run sensitivity analysis for all specified treatments."""
    ensure_dir(OUTPUT_DIR)
    _, df_norm = preprocess_data()
    gml = build_gml_no_u()

    strengths_1d = np.linspace(-MAX_STRENGTH_1D, MAX_STRENGTH_1D, N_POINTS_1D)
    strengths_2d = np.linspace(-MAX_STRENGTH_2D, MAX_STRENGTH_2D, N_POINTS_2D)

    summary_rows = []

    for treatment in TREATMENTS:
        if treatment not in df_norm.columns:
            print(f"[SKIP] {treatment} was not found in the standardized dataframe.")
            continue

        print("\n" + "=" * 80)
        print(f"Treatment: {treatment}")

        model = CausalModel(
            data=df_norm,
            treatment=treatment,
            outcome=OUTCOME,
            graph=gml,
        )

        estimand = model.identify_effect(proceed_when_unidentifiable=True)

        estimate = model.estimate_effect(
            identified_estimand=estimand,
            method_name="backdoor.econml.dml.CausalForestDML",
            method_params={
                "init_params": CF_INIT_PARAMS,
                "fit_params": {},
            },
        )

        base_ate = float(estimate.value)
        print("Base ATE:", base_ate)

        treatment_dir = os.path.join(OUTPUT_DIR, f"treat_{treatment}")
        ensure_dir(treatment_dir)

        save_text(
            os.path.join(treatment_dir, "base_estimand_and_estimate.txt"),
            "===== Estimand =====\n"
            + str(estimand)
            + "\n\n===== Estimate =====\n"
            + str(estimate)
            + "\n",
        )

        line_grid = run_line_grid(
            model=model,
            estimand=estimand,
            estimate=estimate,
            treatment=treatment,
            base_ate=base_ate,
            strengths=strengths_1d,
        )
        line_grid.to_csv(
            os.path.join(treatment_dir, "sensitivity_line_grid.csv"),
            index=False,
        )

        tip = {
            "treatment": treatment,
            "base_ate": base_ate,
            "tipping_strength_t": None,
            "tipping_strength_y": None,
            "tipping_new_ate": None,
        }

        if SAVE_NUMERIC_GRID:
            grid_df, tip = run_numeric_grid(
                model=model,
                estimand=estimand,
                estimate=estimate,
                treatment=treatment,
                base_ate=base_ate,
                strengths=strengths_2d,
            )
            grid_df.to_csv(
                os.path.join(treatment_dir, "sensitivity_numeric_grid.csv"),
                index=False,
            )

        summary_rows.append(tip)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "tipping_points_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\nDone. Outputs saved to:", OUTPUT_DIR)
    print("Tipping-point summary:", summary_path)


if __name__ == "__main__":
    set_all_seeds(SEED)
    main()