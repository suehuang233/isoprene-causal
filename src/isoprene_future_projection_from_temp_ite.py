#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project future isoprene changes from temperature ITE estimates.

This script uses the temperature ITE output from the main DoWhy/EconML analysis,
converts standardized ITE values to raw-unit slopes, constructs a local
Temp -> dIsoprene/dTemp curve, and applies projected May-October temperature
changes to estimate future isoprene changes.

The projection follows this logic:
1. Use the observed-data mean temperature as the local baseline temperature.
2. Use the observed-data mean isoprene as the baseline isoprene level.
3. Use Wang projected temperature changes relative to the Wang baseline,
   rather than using Wang absolute temperatures directly.
4. Correct future temperatures as:
       T_future_corrected = mean(T_observed) + (T_wang_future - T_wang_baseline)
5. Integrate the local ITE-based slope curve from the observed baseline
   temperature to each corrected future temperature.
"""

import os

import numpy as np
import pandas as pd


# =============================================================================
# User configuration
# =============================================================================
OBS_DATA_PATH = "../data/month5to10-5to20hour-2020.csv"

ITE_TEMP_CSV = os.path.join(
    "../results/ite_temp_radiation_curves",
    "ITE_Temp_iters3.csv",
)

# This can be either:
# 1. a precomputed yearly file with columns: Year, RCP, Temp_MayOct_mean; or
# 2. the original Wang-format wide CSV/XLSX, which will be parsed below.
WANG_TEMP_FILE = "wang_yearly_temp_mean_MayOct.csv"

OUT_DIR = "../results/project_future_isoprene_from_ITE"

MONTH_MIN, MONTH_MAX = 5, 10
HOUR_MIN, HOUR_MAX = 5, 20

OUTCOME = "Isoprene"
TREAT = "Temp"

MONTHS_TO_AVG = list(range(5, 11))
BASELINE_YEAR_LABEL = "1996-2018"
BASELINE_OUTPUT_YEAR = "2020"

LOCAL_SLOPE_BIN_WIDTH = 1.0
LOCAL_SLOPE_AGGREGATION = "mean"  # Use "mean" or "median".
INTEGRATION_STEP = 0.25


# =============================================================================
# Utilities
# =============================================================================
def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_obs_data(path: str) -> pd.DataFrame:
    """Load observed data and apply the same Month/Hour filter used for ITE estimation."""
    df = pd.read_csv(path)
    df = df[(df["Month"] >= MONTH_MIN) & (df["Month"] <= MONTH_MAX)]
    df = df[(df["Hour"] >= HOUR_MIN) & (df["Hour"] <= HOUR_MAX)]
    df = df.reset_index(drop=True)

    df = df.replace([np.inf, -np.inf], np.nan)
    if df[[TREAT, OUTCOME]].isna().any().any():
        raise ValueError(
            f"Observed data contain NaN or infinite values in {TREAT} or {OUTCOME}."
        )

    return df


def load_temp_ite(ite_csv: str, nrows_expected: int) -> pd.DataFrame:
    """Load Temp ITE output and calculate the across-iteration mean ITE."""
    ite_df = pd.read_csv(ite_csv)

    if len(ite_df) != nrows_expected:
        raise ValueError(
            f"Row mismatch: observed data have {nrows_expected} rows, "
            f"but the ITE file has {len(ite_df)} rows. Make sure the ITE file "
            "was generated from the same data and filtering window."
        )

    ite_cols = [col for col in ite_df.columns if col.startswith("ITE_Temp_")]
    if not ite_cols:
        raise ValueError(
            "No ITE_Temp_* columns were found. Make sure this is the Temp ITE output."
        )

    ite_df["ITE_Temp_mean"] = ite_df[ite_cols].mean(axis=1)
    return ite_df


def ite_z_to_dydt_raw(df_obs: pd.DataFrame, ite_mean: np.ndarray) -> np.ndarray:
    """Convert standardized Temp ITE values to raw-unit dIsoprene/dTemp slopes."""
    sd_t = df_obs[TREAT].std(ddof=1)
    sd_y = df_obs[OUTCOME].std(ddof=1)

    if sd_t == 0 or sd_y == 0:
        raise ValueError(
            "The treatment or outcome has zero standard deviation, so raw-unit "
            "slope conversion is undefined."
        )

    return ite_mean * (sd_y / sd_t)


# =============================================================================
# Wang temperature input
# =============================================================================
def read_wang_temperature_raw(file_path: str) -> pd.DataFrame:
    """
    Parse the original Wang-format temperature file.

    Expected format: the first seven columns contain Year, Month, Temp blocks
    for RCP4.5, RCP6.0, and RCP8.5.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        raw = pd.read_csv(file_path, low_memory=False)
    elif ext in [".xlsx", ".xls"]:
        raw = pd.read_excel(file_path)
    else:
        raise ValueError("Wang temperature input must be a .csv, .xlsx, or .xls file.")

    raw = raw.iloc[:, :7].copy()
    raw.columns = [
        "Year", "Month_45", "Temp_45",
        "Month_60", "Temp_60",
        "Month_85", "Temp_85",
    ]

    raw["Year"] = raw["Year"].astype(str).str.strip()
    raw = raw[raw["Year"].str.lower() != "year"].copy()
    raw["Year"] = raw["Year"].replace(["nan", "None", ""], np.nan).ffill()

    for col in ["Month_45", "Month_60", "Month_85"]:
        raw[col] = pd.to_numeric(raw[col].astype(str).str.strip(), errors="coerce")

    for col in ["Temp_45", "Temp_60", "Temp_85"]:
        raw[col] = pd.to_numeric(raw[col].astype(str).str.strip(), errors="coerce")

    df45 = raw[["Year", "Month_45", "Temp_45"]].rename(
        columns={"Month_45": "Month", "Temp_45": "Temp"}
    )
    df45["RCP"] = "RCP4.5"

    df60 = raw[["Year", "Month_60", "Temp_60"]].rename(
        columns={"Month_60": "Month", "Temp_60": "Temp"}
    )
    df60["RCP"] = "RCP6.0"

    df85 = raw[["Year", "Month_85", "Temp_85"]].rename(
        columns={"Month_85": "Month", "Temp_85": "Temp"}
    )
    df85["RCP"] = "RCP8.5"

    out = pd.concat([df45, df60, df85], ignore_index=True)
    out = out.dropna(subset=["Year", "Month", "Temp"]).copy()
    out["Year"] = out["Year"].astype(str).str.strip()
    out["Month"] = out["Month"].astype(int)

    return out


def compute_yearly_temp_mean(
    wang_long: pd.DataFrame,
    months_to_avg: list[int],
) -> pd.DataFrame:
    """Calculate May-October mean temperature for each Year x RCP combination."""
    df = wang_long[wang_long["Month"].isin(months_to_avg)].copy()

    temp_mean = (
        df.groupby(["Year", "RCP"], as_index=False)["Temp"]
        .mean()
        .rename(columns={"Temp": "Temp_MayOct_mean"})
    )

    return temp_mean


def load_projected_temperature(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Load projected temperature data.

    If the file already contains Year, RCP, and Temp_MayOct_mean, it is used
    directly. Otherwise, the original Wang-format wide file is parsed and then
    averaged over MONTHS_TO_AVG.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        raw = pd.read_csv(file_path, low_memory=False)
    elif ext in [".xlsx", ".xls"]:
        raw = pd.read_excel(file_path)
    else:
        raise ValueError("Projected temperature input must be a .csv, .xlsx, or .xls file.")

    required = {"Year", "RCP", "Temp_MayOct_mean"}
    if required.issubset(raw.columns):
        temp_mean = raw[["Year", "RCP", "Temp_MayOct_mean"]].copy()
        temp_mean["Year"] = temp_mean["Year"].astype(str).str.strip()
        temp_mean["RCP"] = temp_mean["RCP"].astype(str).str.strip()
        temp_mean["Temp_MayOct_mean"] = pd.to_numeric(
            temp_mean["Temp_MayOct_mean"],
            errors="coerce",
        )
        temp_mean = temp_mean.dropna(subset=["Year", "RCP", "Temp_MayOct_mean"])
        return temp_mean.reset_index(drop=True), None

    wang_long = read_wang_temperature_raw(file_path)
    temp_mean = compute_yearly_temp_mean(wang_long, MONTHS_TO_AVG)
    return temp_mean, wang_long


def find_baseline_temp(
    temp_mean: pd.DataFrame,
    baseline_year_label: str = BASELINE_YEAR_LABEL,
) -> float:
    """Find the Wang baseline May-October temperature."""
    temp_mean = temp_mean.copy()
    temp_mean["Year_clean"] = temp_mean["Year"].astype(str).str.strip()

    base_rows = temp_mean[temp_mean["Year_clean"] == baseline_year_label].copy()
    if base_rows.empty:
        available_years = temp_mean["Year_clean"].unique()[:20].tolist()
        raise ValueError(
            f"Baseline year label '{baseline_year_label}' was not found. "
            f"Available Year examples: {available_years}"
        )

    return float(base_rows["Temp_MayOct_mean"].mean())


# =============================================================================
# Local ITE curve and projection
# =============================================================================
def build_local_slope_curve(
    df_obs: pd.DataFrame,
    ite_df: pd.DataFrame,
    months_to_avg: list[int] | None = None,
    bin_width: float = LOCAL_SLOPE_BIN_WIDTH,
    aggregation: str = LOCAL_SLOPE_AGGREGATION,
) -> pd.DataFrame:
    """Build a local temperature-binned dIsoprene/dTemp curve in raw units."""
    if aggregation not in {"mean", "median"}:
        raise ValueError("aggregation must be either 'mean' or 'median'.")

    dydt_raw = ite_z_to_dydt_raw(df_obs, ite_df["ITE_Temp_mean"].values)

    df_tmp = df_obs[[TREAT, "Month"]].copy()
    df_tmp["dYdT_perC_raw"] = dydt_raw

    if months_to_avg is not None:
        df_tmp = df_tmp[df_tmp["Month"].isin(months_to_avg)].copy()

    df_tmp["Temp_bin"] = np.floor(df_tmp[TREAT] / bin_width) * bin_width

    slope_curve = (
        df_tmp.groupby("Temp_bin", as_index=False)["dYdT_perC_raw"]
        .agg(dYdT=aggregation, n_samples="count")
        .sort_values("Temp_bin")
        .reset_index(drop=True)
    )

    return slope_curve


def integrate_delta_isoprene(
    t0: float,
    t1: float,
    slope_curve: pd.DataFrame,
    step: float = INTEGRATION_STEP,
) -> float:
    """Integrate the local dIsoprene/dTemp curve between two temperatures."""
    if t0 == t1:
        return 0.0

    sign = 1.0
    if t1 < t0:
        t0, t1 = t1, t0
        sign = -1.0

    x = slope_curve["Temp_bin"].values.astype(float)
    y = slope_curve["dYdT"].values.astype(float)

    if len(x) < 2:
        raise ValueError("At least two temperature bins are required for integration.")

    def local_slope(temp: float) -> float:
        temp = np.clip(temp, x.min(), x.max())
        return float(np.interp(temp, x, y))

    grid = np.arange(t0, t1 + 1e-9, step)
    if len(grid) == 1 or grid[-1] < t1:
        grid = np.append(grid, t1)

    values = np.array([local_slope(temp) for temp in grid])

    if hasattr(np, "trapezoid"):
        delta = np.trapezoid(values, grid)
    else:
        delta = np.trapz(values, grid)

    return sign * float(delta)


def project_isoprene_change_local_ite(
    temp_mean: pd.DataFrame,
    t_base: float,
    y_base: float,
    slope_curve: pd.DataFrame,
) -> pd.DataFrame:
    """Project future isoprene changes using local ITE integration."""
    out = temp_mean.copy()
    out["T_base"] = t_base
    out["Isoprene_base"] = y_base

    out["Delta_T_vs_base"] = out["Temp_MayOct_mean"] - t_base
    out["Delta_Isoprene_vs_base"] = [
        integrate_delta_isoprene(t_base, float(temp), slope_curve)
        for temp in out["Temp_MayOct_mean"].values
    ]
    out["Isoprene_future"] = out["Isoprene_base"] + out["Delta_Isoprene_vs_base"]
    out["Isoprene_pct_change (%)"] = (
        out["Delta_Isoprene_vs_base"] / out["Isoprene_base"]
    ) * 100

    return out


def build_baseline_rows(rcps: list[str], t_base: float, y_base: float) -> pd.DataFrame:
    """Create baseline rows for the observed reference year."""
    return pd.DataFrame(
        {
            "Year": [BASELINE_OUTPUT_YEAR] * len(rcps),
            "RCP": rcps,
            "Temp_MayOct_mean": [t_base] * len(rcps),
            "T_base": [t_base] * len(rcps),
            "Isoprene_base": [y_base] * len(rcps),
            "Delta_T_vs_base": [0.0] * len(rcps),
            "Delta_Isoprene_vs_base": [0.0] * len(rcps),
            "Isoprene_future": [y_base] * len(rcps),
            "Isoprene_pct_change (%)": [0.0] * len(rcps),
        }
    )


# =============================================================================
# Main workflow
# =============================================================================
def main() -> None:
    """Run the future isoprene projection from temperature ITE estimates."""
    ensure_dir(OUT_DIR)

    df_obs = load_obs_data(OBS_DATA_PATH)
    ite_df = load_temp_ite(ITE_TEMP_CSV, nrows_expected=len(df_obs))

    dydt = ite_z_to_dydt_raw(df_obs, ite_df["ITE_Temp_mean"].values)
    theta_mean = float(np.mean(dydt))
    theta_median = float(np.median(dydt))

    rowwise = df_obs[[TREAT, OUTCOME, "Month", "Hour"]].copy()
    rowwise["ITE_Temp_mean_z"] = ite_df["ITE_Temp_mean"].values
    rowwise["dYdT_perC_raw"] = dydt
    rowwise_path = os.path.join(OUT_DIR, "Temp_ITE_rowwise_dYdT.csv")
    rowwise.to_csv(rowwise_path, index=False)

    temp_mean, wang_long = load_projected_temperature(WANG_TEMP_FILE)

    if wang_long is not None:
        wang_long_path = os.path.join(OUT_DIR, "wang_temperature_long_clean.csv")
        wang_long.to_csv(wang_long_path, index=False)
    else:
        wang_long_path = None

    yearly_temp_path = os.path.join(OUT_DIR, "wang_yearly_temp_mean_MayOct.csv")
    temp_mean.to_csv(yearly_temp_path, index=False)

    wang_baseline_temp = find_baseline_temp(
        temp_mean,
        baseline_year_label=BASELINE_YEAR_LABEL,
    )

    slope_curve = build_local_slope_curve(
        df_obs=df_obs,
        ite_df=ite_df,
        months_to_avg=MONTHS_TO_AVG,
        bin_width=LOCAL_SLOPE_BIN_WIDTH,
        aggregation=LOCAL_SLOPE_AGGREGATION,
    )
    slope_curve_path = os.path.join(OUT_DIR, "Temp_local_slope_curve_1C.csv")
    slope_curve.to_csv(slope_curve_path, index=False)

    observed_temp_base = float(df_obs[TREAT].mean())
    observed_isoprene_base = float(df_obs[OUTCOME].mean())

    temp_mean_corrected = temp_mean.copy()
    temp_mean_corrected["Temp_MayOct_mean"] = observed_temp_base + (
        temp_mean_corrected["Temp_MayOct_mean"] - wang_baseline_temp
    )

    temp_mean_corrected = temp_mean_corrected[
        temp_mean_corrected["Year"].astype(str).str.strip() != BASELINE_YEAR_LABEL
    ].copy()

    rcp_order = ["RCP4.5", "RCP6.0", "RCP8.5"]
    available_rcps = temp_mean["RCP"].astype(str).str.strip().unique().tolist()
    rcps = [rcp for rcp in rcp_order if rcp in available_rcps]
    if not rcps:
        rcps = available_rcps

    baseline_rows = build_baseline_rows(
        rcps=rcps,
        t_base=observed_temp_base,
        y_base=observed_isoprene_base,
    )

    projection = project_isoprene_change_local_ite(
        temp_mean=temp_mean_corrected,
        t_base=observed_temp_base,
        y_base=observed_isoprene_base,
        slope_curve=slope_curve,
    )

    projection = pd.concat([baseline_rows, projection], ignore_index=True)

    projection_path = os.path.join(
        OUT_DIR,
        "future_isoprene_projection_yearly_localITE.csv",
    )
    projection.to_csv(projection_path, index=False)

    print("\nDone.")
    print("Rowwise dY/dT:", rowwise_path)
    if wang_long_path is not None:
        print("Cleaned Wang long table:", wang_long_path)
    print("Yearly Wang temperature:", yearly_temp_path)
    print("Local ITE slope curve:", slope_curve_path)
    print("Future isoprene projection:", projection_path)
    print("Observed baseline Temp:", observed_temp_base)
    print("Observed baseline Isoprene:", observed_isoprene_base)
    print("Wang baseline Temp:", wang_baseline_temp)
    print("theta_mean dY/dT:", theta_mean)
    print("theta_median dY/dT:", theta_median)


if __name__ == "__main__":
    main()