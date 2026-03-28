import os
import random

import numpy as np
import pandas as pd

from common import (
    ensure_dir,
    normalize_data_zscore,
    plot_global_dag,
    save_results_to_file,
    set_all_seeds,
    tune_hyperparameters,
    estimate_causal_ite_dowhy,
)

# =========================
# User config
# =========================
SEED = 2026
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = "./data/month5to10-5to20hour-2020.csv"
OUTPUT_DIR = "./isoprene_dowhy_causaleffect/withoutOH"

# Analysis window
MONTH_MIN, MONTH_MAX = 5, 10
HOUR_MIN, HOUR_MAX = 5, 20

# Modeling variables
OUTCOME = "Isoprene"
TREATMENTS = ["RH", "Temp", "Radiation", "u", "v", "Oxides", "Toluene", "leaf_stage"]
DROP_COLS = ["Year", "WS", "WD", "LAI"]

ONLY_RUN = []   # [] means run all treatments; e.g., ["Temp"] means run only Temp

ITERATIONS = 3
REFUTE_THRESHOLD = 0.1

# DAG plotting
PLOT_DAG = True


# =========================
# Main
# =========================
def main():
    set_all_seeds(SEED)
    ensure_dir(OUTPUT_DIR)

    # --- DAG ---
    # Keep the DAG exactly in the original direct style

    nodes_label = ['RH', 'Temp', 'Radiation', 'u', 'v', 'Oxides', 'Toluene', 'leaf_stage', 'U', 'Isoprene']
    nodes =       ['A',   'B',      'C',      'D', 'E', 'F',      'H',       'I',          'U', 'Y']

    edges = [
        'AY', 'BY', 'CY', 'DY', 'EY', 'FY', 'HY', 'IY',
        'AF', 'AH',       # RH -> Oxides / Toluene
        'BA', 'BH', 'BF', # Temp -> RH / Toluene / Oxides
        'CH', 'CF',       # Radiation -> Toluene / Oxides
        'DF', 'DH',       # u -> Oxides / Toluene
        'EF', 'EH',       # v -> Oxides / Toluene
        'HF',             # Toluene -> Oxides
        'UY'
    ]

    gml_string = 'graph [directed 1\n'
    for node, label in zip(nodes, nodes_label):
        gml_string += f'\tnode [id "{node}" label "{label}"]\n'
    for e in edges:
        gml_string += f'\tedge [source "{e[0]}" target "{e[1]}"]\n'
    gml_string += ']'

    if PLOT_DAG:
        plot_global_dag(gml_string, os.path.join(OUTPUT_DIR, "Global_DAG_without_OH.png"))

    # --- Read data ---
    df = pd.read_csv(DATA_PATH)

    # Basic filters
    df = df[(df["Month"] >= MONTH_MIN) & (df["Month"] <= MONTH_MAX)]
    df = df[(df["Hour"] >= HOUR_MIN) & (df["Hour"] <= HOUR_MAX)]

    df = df.drop(columns=DROP_COLS, errors="ignore").reset_index(drop=True)

    # Keep only required columns
    needed = ["Month", "Hour", OUTCOME] + TREATMENTS
    needed = [c for c in needed if c in df.columns]
    df = df[needed].copy()

    # Drop invalid rows
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[OUTCOME] + [t for t in TREATMENTS if t in df.columns]).reset_index(drop=True)

    # Z-score standardization
    exclude = ["Month", "Hour"]
    df_norm = normalize_data_zscore(df, exclude_columns=exclude)

    controls_used = []
    ate_rows = []
    run_list = TREATMENTS if (not ONLY_RUN) else ONLY_RUN

    for treat in run_list:
        if treat not in df_norm.columns:
            print(f"[SKIP] {treat} not found in dataframe.")
            continue

        print("\n" + "=" * 80)
        print(f"Treatment: {treat}")

        # Keep the original logic for covariate selection
        nodes_label_model = ['RH', 'Temp', 'Radiation', 'u', 'v', 'Oxides', 'Toluene', 'leaf_stage', 'U', 'Isoprene']

        nodes_label_0 = nodes_label_model.copy()
        nodes_label_0.remove("U")
        for var in [treat, OUTCOME]:
            if var in nodes_label_0:
                nodes_label_0.remove(var)
        covariates_for_tune = nodes_label_0

        best_params_y, best_params_t = tune_hyperparameters(
            df_norm, covariates_for_tune, treat, OUTCOME, SEED
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
                REFUTE_THRESHOLD,
                SEED + i,
            )

            warn_any = warn_any or warn
            ate_list.append(ate)
            ite_df[f"ITE_{i+1}"] = ite

            if i == 0:
                controls_used.append(
                    pd.DataFrame({
                        "treatment": [treat],
                        "controls_used_by_DoWhy": [", ".join(cov_used)]
                    })
                )

                save_results_to_file(
                    estimand,
                    ref_subset,
                    ref_rcc,
                    best_params_y,
                    best_params_t,
                    os.path.join(OUTPUT_DIR, f"{treat}_results.txt"),
                )

        ite_df["ITE_mean"] = ite_df[[c for c in ite_df.columns if c.startswith("ITE_")]].mean(axis=1)
        ite_df.to_csv(os.path.join(OUTPUT_DIR, f"{treat}_ITE.csv"), index=False)

        ate_mean = float(np.mean(ate_list))
        ate_std = float(np.std(ate_list, ddof=1)) if len(ate_list) > 1 else 0.0

        ate_rows.append({
            "treatment": treat,
            "ATE_mean": ate_mean,
            "ATE_std": ate_std,
            "warning_flag": warn_any,
        })

    if ate_rows:
        pd.DataFrame(ate_rows).to_csv(os.path.join(OUTPUT_DIR, "ATE_summary.csv"), index=False)

    if controls_used:
        pd.concat(controls_used, ignore_index=True).to_csv(
            os.path.join(OUTPUT_DIR, "controls_used.csv"), index=False
        )

    print("\nFinished: without OH analysis completed.")


if __name__ == "__main__":
    main()
