#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocess ITE outputs to reconstruct temperature- and radiation-isoprene curves.

This script converts standardized ITE estimates from the main DoWhy/EconML analysis
back to raw-unit slopes and produces binned CSV summaries for Temp and Radiation.

The row order of the input data must match the row order used when creating the
corresponding ITE CSV files.
"""

import os

import numpy as np
import pandas as pd


# =============================================================================
# User configuration
# =============================================================================
# Place the input CSV in the same directory as this script, or replace this path.
DATA_PATH = "../data/month5to10-5to20hour-2020.csv"

BASE_OUTPUT_DIR = "../results/causal_effects"

MONTH_MIN, MONTH_MAX = 5, 10
HOUR_MIN, HOUR_MAX = 5, 20

OUTCOME = "Isoprene"

TREATMENT_CONFIGS = [
    {
        "treatment": "Temp",
        "ite_csv": os.path.join(BASE_OUTPUT_DIR, "ITE_Temp_iters3.csv"),
        "output_dir": os.path.join(BASE_OUTPUT_DIR, "postprocess_temp_curve"),
        "bin_width_raw": 1.0,
        "slope_column": "dYdT_perC",
        "curve_csv": "Temp_curve_binned_1C.csv",
        "rowwise_csv": "Temp_ITE_to_raw_isoprene_rowwise.csv",
        "save_rowwise": False,
    },
    {
        "treatment": "Radiation",
        "ite_csv": os.path.join(BASE_OUTPUT_DIR, "ITE_Radiation_iters3.csv"),
        "output_dir": os.path.join(BASE_OUTPUT_DIR, "postprocess_Radiation_curve"),
        "bin_width_raw": 50.0,
        "slope_column": "dYdRad_perUnit",
        "curve_csv": "Radiation_curve_binned.csv",
        "rowwise_csv": "Radiation_ITE_to_raw_isoprene_rowwise.csv",
        "save_rowwise": True,
    },
]


# =============================================================================
# Utilities
# =============================================================================
def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_filtered_data(data_path: str) -> pd.DataFrame:
    """Load the raw data and apply the same Month/Hour filter used in the main model."""
    df = pd.read_csv(data_path)
    df = df[(df["Month"] >= MONTH_MIN) & (df["Month"] <= MONTH_MAX)]
    df = df[(df["Hour"] >= HOUR_MIN) & (df["Hour"] <= HOUR_MAX)]
    return df.reset_index(drop=True)


def get_ite_mean(ite_df: pd.DataFrame, treatment: str) -> pd.Series:
    """Average ITE estimates across iterations for one treatment."""
    ite_cols = [col for col in ite_df.columns if col.startswith(f"ITE_{treatment}_")]
    if not ite_cols:
        raise ValueError(
            f"No ITE columns were found for {treatment}. "
            f"Expected columns with prefix 'ITE_{treatment}_'."
        )
    return ite_df[ite_cols].mean(axis=1)


def build_rowwise_output(
    df: pd.DataFrame,
    ite_df: pd.DataFrame,
    treatment: str,
    slope_column: str,
) -> pd.DataFrame:
    """Convert standardized ITE values to raw-unit slopes and rowwise predictions."""
    if len(df) != len(ite_df):
        raise ValueError(
            f"Row mismatch for {treatment}: raw data has {len(df)} rows, "
            f"but the ITE file has {len(ite_df)} rows."
        )

    required_columns = [treatment, OUTCOME, "Month", "Hour"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in raw data: {missing_columns}")

    ite_mean_col = f"ITE_{treatment}_mean"
    treatment_z_col = f"{treatment}_z"

    df_out = df[required_columns].copy()
    df_out[ite_mean_col] = get_ite_mean(ite_df, treatment).values

    mu_t = df[treatment].mean()
    sd_t = df[treatment].std(ddof=1)
    mu_y = df[OUTCOME].mean()
    sd_y = df[OUTCOME].std(ddof=1)

    if sd_t == 0 or sd_y == 0:
        raise ValueError(
            f"Cannot convert standardized ITE values for {treatment}: "
            "the treatment or outcome has zero standard deviation."
        )

    df_out[treatment_z_col] = (df_out[treatment] - mu_t) / sd_t
    df_out["Isoprene_z"] = (df_out[OUTCOME] - mu_y) / sd_y

    df_out["Yhat_z"] = df_out[ite_mean_col] * df_out[treatment_z_col]
    df_out[slope_column] = df_out[ite_mean_col] * (sd_y / sd_t)
    df_out["Yhat_raw"] = mu_y + df_out[slope_column] * (df_out[treatment] - mu_t)

    return df_out


def build_binned_curve(
    df_out: pd.DataFrame,
    treatment: str,
    slope_column: str,
    bin_width_raw: float,
) -> pd.DataFrame:
    """Aggregate rowwise predictions into raw-treatment bins."""
    treatment_z_col = f"{treatment}_z"
    ite_mean_col = f"ITE_{treatment}_mean"
    bin_col = f"{treatment}_bin_raw"

    df_out = df_out.copy()
    df_out[bin_col] = np.floor(df_out[treatment] / bin_width_raw) * bin_width_raw

    curve = (
        df_out.groupby(bin_col, as_index=False)
        .agg(
            **{
                f"{treatment}_raw_mean": (treatment, "mean"),
                f"{treatment}_z_mean": (treatment_z_col, "mean"),
                "Y_obs_raw_mean": (OUTCOME, "mean"),
                "Y_obs_z_mean": ("Isoprene_z", "mean"),
                "Y_hat_raw_mean": ("Yhat_raw", "mean"),
                "Y_hat_raw_std": ("Yhat_raw", "std"),
                "Y_hat_z_mean": ("Yhat_z", "mean"),
                "Y_hat_z_std": ("Yhat_z", "std"),
                f"{slope_column}_mean": (slope_column, "mean"),
                "ITE_z_mean": (ite_mean_col, "mean"),
                "n": ("Yhat_raw", "size"),
            }
        )
        .sort_values(bin_col)
    )

    return curve


def postprocess_treatment(df: pd.DataFrame, config: dict) -> None:
    """Run rowwise conversion and binned-curve aggregation for one treatment."""
    treatment = config["treatment"]
    output_dir = config["output_dir"]
    ensure_dir(output_dir)

    ite_df = pd.read_csv(config["ite_csv"])

    df_out = build_rowwise_output(
        df=df,
        ite_df=ite_df,
        treatment=treatment,
        slope_column=config["slope_column"],
    )

    if config["save_rowwise"]:
        rowwise_path = os.path.join(output_dir, config["rowwise_csv"])
        df_out.to_csv(rowwise_path, index=False)
        print("Saved rowwise output:", rowwise_path)

    curve = build_binned_curve(
        df_out=df_out,
        treatment=treatment,
        slope_column=config["slope_column"],
        bin_width_raw=config["bin_width_raw"],
    )

    curve_path = os.path.join(output_dir, config["curve_csv"])
    curve.to_csv(curve_path, index=False)
    print("Saved binned curve:", curve_path)


def main() -> None:
    """Run Temp and Radiation postprocessing."""
    df = load_filtered_data(DATA_PATH)

    for config in TREATMENT_CONFIGS:
        print("\n" + "=" * 80)
        print(f"Postprocessing treatment: {config['treatment']}")
        postprocess_treatment(df=df, config=config)


if __name__ == "__main__":
    main()
