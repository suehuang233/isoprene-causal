import os
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from dowhy import CausalModel


# =========================
# Configuration
# =========================
SEED = 2026
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = "./data/month5to10-5to20hour-2020.csv"
OUTPUT_DIR = "./isoprene_dowhy_causaleffect/sensitivity_analysis"

MONTH_MIN, MONTH_MAX = 5, 10
HOUR_MIN, HOUR_MAX = 5, 20

OUTCOME = "Isoprene"
TREATMENTS = ["RH", "Temp", "Radiation", "u", "v", "Oxides", "Toluene", "leaf_stage", "OH"]
DROP_COLS = ["Year", "WS", "WD", "LAI"]

# Sensitivity-strength grids
# These coefficients are used in direct-simulation with linear confounding effects.
# Because the analysis is performed on standardized variables, the strengths are in SD units.
MAX_STRENGTH_1D = 1.0
N_POINTS_1D = 41

MAX_STRENGTH_2D = 1.0
N_POINTS_2D = 41

# Whether to save the full 2D numeric sensitivity grid
SAVE_NUMERIC_GRID = True

# Estimator parameters
CF_INIT_PARAMS = dict(
    model_y=LGBMRegressor(random_state=SEED),
    model_t=LGBMRegressor(random_state=SEED),
    n_estimators=500,
    min_var_fraction_leaf=0.1,
    random_state=SEED,
)


# =========================
# Utilities
# =========================
def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def normalize_data_zscore(df: pd.DataFrame, exclude_columns=None) -> pd.DataFrame:
    """
    Apply z-score standardization to all columns except exclude_columns.
    """
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = []

    cols = [c for c in df.columns if c not in exclude_columns]
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


# =========================
# DAG
# Keep the DAG in direct form
# The explicit unobserved node U is removed here, consistent with the uploaded sensitivity script
# =========================
def build_gml_no_u() -> str:
    nodes_label = [
        "RH", "Temp", "Radiation", "u", "v",
        "Oxides", "OH", "Toluene", "leaf_stage", "Isoprene"
    ]
    nodes = [
        "A", "B", "C", "D", "E",
        "F", "G", "H", "I", "Y"
    ]

    edges = [
        # Direct effects on outcome
        "AY", "BY", "CY", "DY", "EY", "FY", "HY", "GY", "IY",

        # RH -> Oxides / OH / Toluene
        "AF", "AG", "AH",

        # Temp -> RH / Toluene / Oxides / OH
        "BA", "BH", "BF", "BG",

        # Radiation -> Toluene / Oxides / OH
        "CH", "CF", "CG",

        # u -> Oxides / OH / Toluene
        "DF", "DG", "DH",

        # v -> Oxides / OH / Toluene
        "EF", "EG", "EH",

        # OH -> Oxides / Toluene
        "GF", "GH",

        # Toluene -> Oxides
        "HF",
    ]

    gml = 'graph [directed 1\n'
    for node, label in zip(nodes, nodes_label):
        gml += f'\tnode [id "{node}" label "{label}"]\n'
    for edge in edges:
        gml += f'\tedge [source "{edge[0]}" target "{edge[1]}"]\n'
    gml += ']'
    return gml


# =========================
# Data preprocessing
# Match the main analysis workflow
# =========================
def preprocess_like_main():
    df = pd.read_csv(DATA_PATH)

    # Apply month and hour filters
    df = df[(df["Month"] >= MONTH_MIN) & (df["Month"] <= MONTH_MAX)]
    df = df[(df["Hour"] >= HOUR_MIN) & (df["Hour"] <= HOUR_MAX)]

    # Drop unused columns
    df = df.drop(columns=DROP_COLS, errors="ignore").reset_index(drop=True)

    # Keep required columns only
    needed = ["Month", "Hour", OUTCOME] + TREATMENTS
    needed = [c for c in needed if c in df.columns]
    df = df[needed].copy()

    # Complete-case filtering
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[OUTCOME] + [t for t in TREATMENTS if t in df.columns]).reset_index(drop=True)

    # Standardize all variables except Month and Hour
    df_norm = normalize_data_zscore(df, exclude_columns=["Month", "Hour"])
    return df, df_norm


# =========================
# One-dimensional sensitivity scan
# Vary treatment-side confounding only, keep outcome-side confounding at zero
# =========================
def run_1d_sensitivity_scan(model, estimand, estimate, treat, base_ate, strengths):
    rows = []

    for a in strengths:
        ref = model.refute_estimate(
            estimand,
            estimate,
            method_name="add_unobserved_common_cause",
            simulation_method="direct-simulation",
            confounders_effect_on_treatment="linear",
            confounders_effect_on_outcome="linear",
            effect_strength_on_treatment=float(a),
            effect_strength_on_outcome=0.0,
            random_seed=SEED,
            plotmethod=None,
        )

        new_ate = float(ref.new_effect)
        rows.append({
            "treatment": treat,
            "base_ate": float(base_ate),
            "strength_t": float(a),
            "strength_y": 0.0,
            "new_ate": new_ate,
            "effect_difference": float(ref.effect_difference)
            if hasattr(ref, "effect_difference")
            else (new_ate - base_ate),
        })

    return pd.DataFrame(rows)


# =========================
# Two-dimensional numeric sensitivity grid
# Used for tipping-point identification
# =========================
def run_2d_numeric_grid(model, estimand, estimate, treat, base_ate, strengths):
    rows = []

    for a in strengths:
        for b in strengths:
            ref = model.refute_estimate(
                estimand,
                estimate,
                method_name="add_unobserved_common_cause",
                simulation_method="direct-simulation",
                confounders_effect_on_treatment="linear",
                confounders_effect_on_outcome="linear",
                effect_strength_on_treatment=float(a),
                effect_strength_on_outcome=float(b),
                random_seed=SEED,
                plotmethod=None,
            )

            new_ate = float(ref.new_effect)
            rows.append({
                "treatment": treat,
                "base_ate": float(base_ate),
                "strength_t": float(a),
                "strength_y": float(b),
                "new_ate": new_ate,
                "effect_difference": float(ref.effect_difference)
                if hasattr(ref, "effect_difference")
                else (new_ate - base_ate),
            })

    grid = pd.DataFrame(rows)

    # Define the tipping point as the smallest L2 norm where the estimated effect changes sign
    if base_ate > 0:
        cand = grid[grid["new_ate"] <= 0].copy()
    else:
        cand = grid[grid["new_ate"] >= 0].copy()

    if len(cand) > 0:
        cand["norm"] = np.sqrt(cand["strength_t"] ** 2 + cand["strength_y"] ** 2)
        best = cand.sort_values("norm").iloc[0]
        tip = {
            "treatment": treat,
            "base_ate": float(base_ate),
            "tipping_strength_t": float(best["strength_t"]),
            "tipping_strength_y": float(best["strength_y"]),
            "tipping_new_ate": float(best["new_ate"]),
        }
    else:
        tip = {
            "treatment": treat,
            "base_ate": float(base_ate),
            "tipping_strength_t": None,
            "tipping_strength_y": None,
            "tipping_new_ate": None,
        }

    return grid, tip


# =========================
# Main
# =========================
def main():
    ensure_dir(OUTPUT_DIR)
    _, df_norm = preprocess_like_main()
    gml = build_gml_no_u()

    strengths_1d = np.linspace(-MAX_STRENGTH_1D, MAX_STRENGTH_1D, N_POINTS_1D)
    strengths_2d = np.linspace(-MAX_STRENGTH_2D, MAX_STRENGTH_2D, N_POINTS_2D)

    summary_rows = []

    for treat in TREATMENTS:
        if treat not in df_norm.columns:
            print(f"[SKIP] {treat} not found in the standardized dataframe.")
            continue

        print("\n" + "=" * 80)
        print(f"Treatment: {treat}")

        model = CausalModel(
            data=df_norm,
            treatment=treat,
            outcome=OUTCOME,
            graph=gml,
        )

        estimand = model.identify_effect(proceed_when_unidentifiable=True)

        estimate = model.estimate_effect(
            identified_estimand=estimand,
            method_name="backdoor.econml.dml.CausalForestDML",
            method_params={
                "init_params": CF_INIT_PARAMS,
                "fit_params": {}
            }
        )

        base_ate = float(estimate.value)
        print("Base ATE:", base_ate)

        treat_dir = os.path.join(OUTPUT_DIR, f"treat_{treat}")
        ensure_dir(treat_dir)

        # Save estimand and estimate
        with open(os.path.join(treat_dir, "base_estimand_and_estimate.txt"), "w", encoding="utf-8") as f:
            f.write("===== Estimand =====\n")
            f.write(str(estimand) + "\n\n")
            f.write("===== Estimate =====\n")
            f.write(str(estimate) + "\n")

        # Save one-dimensional numeric scan
        scan_1d = run_1d_sensitivity_scan(model, estimand, estimate, treat, base_ate, strengths_1d)
        scan_1d.to_csv(os.path.join(treat_dir, "sensitivity_1d_scan.csv"), index=False)

        # Save a text version of the zero-outcome-confounding refuter at baseline
        ref_line_text = model.refute_estimate(
            estimand,
            estimate,
            method_name="add_unobserved_common_cause",
            simulation_method="direct-simulation",
            confounders_effect_on_treatment="linear",
            confounders_effect_on_outcome="linear",
            effect_strength_on_treatment=0.0,
            effect_strength_on_outcome=0.0,
            random_seed=SEED,
            plotmethod=None,
        )
        with open(os.path.join(treat_dir, "refuter_baseline.txt"), "w", encoding="utf-8") as f:
            f.write(str(ref_line_text) + "\n")

        tip = {
            "treatment": treat,
            "base_ate": base_ate,
            "tipping_strength_t": None,
            "tipping_strength_y": None,
            "tipping_new_ate": None,
        }

        if SAVE_NUMERIC_GRID:
            grid_df, tip = run_2d_numeric_grid(model, estimand, estimate, treat, base_ate, strengths_2d)
            grid_df.to_csv(os.path.join(treat_dir, "sensitivity_numeric_grid.csv"), index=False)

        summary_rows.append(tip)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "tipping_points_summary.csv"), index=False)

    print("\nFinished.")
    print("Output directory:", OUTPUT_DIR)
    print("Saved summary:", os.path.join(OUTPUT_DIR, "tipping_points_summary.csv"))


if __name__ == "__main__":
    set_all_seeds(SEED)
    main()
