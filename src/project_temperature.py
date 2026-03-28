import os
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================
DATA_PATH = "./data/month5to10-5to20hour-2020.csv"

# Temp ITE output from the main DoWhy pipeline
# Expected unified format: Temp_ITE.csv with columns:
# Temp, ITE_1, ITE_2, ..., ITE_mean
ITE_TEMP_CSV = "./isoprene_dowhy_causaleffect/Temp_ITE.csv"

# Cleaned Wang temperature file prepared in advance
# Required columns: Year, RCP, Temp_MayOct_mean
WANG_TEMP_CLEAN_FILE = "./data/wang_yearly_temp_mean_MayOct.csv"

OUT_DIR = "./isoprene_dowhy_causaleffect/project_temperature"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Analysis window
# Must be consistent with the ITE generation window
# ============================================================
MONTH_MIN, MONTH_MAX = 5, 10
HOUR_MIN, HOUR_MAX = 5, 20

OUTCOME = "Isoprene"
TREAT = "Temp"

# Baseline label in the cleaned Wang temperature file
BASELINE_YEAR_LABEL = "1996-2018"


# ============================================================
# Read observational data
# The row order must remain aligned with the ITE file
# ============================================================
def load_obs_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df[(df["Month"] >= MONTH_MIN) & (df["Month"] <= MONTH_MAX)]
    df = df[(df["Hour"] >= HOUR_MIN) & (df["Hour"] <= HOUR_MAX)]
    df = df.reset_index(drop=True)

    df = df.replace([np.inf, -np.inf], np.nan)
    if df[[TREAT, OUTCOME]].isna().any().any():
        raise ValueError(
            "The observational dataset contains NaN or inf values in Temp or Isoprene. "
            "Please clean the data first."
        )

    return df


# ============================================================
# Read Temp ITE file and compute ITE_Temp_mean if needed
# ============================================================
def load_temp_ite(ite_csv: str, nrows_expected: int) -> pd.DataFrame:
    ite_df = pd.read_csv(ite_csv)

    if len(ite_df) != nrows_expected:
        raise ValueError(
            f"Row mismatch: observational df={nrows_expected}, ITE file={len(ite_df)}. "
            "Please make sure the Temp ITE file was generated from the same dataset "
            "and the same filtering window."
        )

    ite_cols = [c for c in ite_df.columns if c.startswith("ITE_") and c != "ITE_mean"]

    # Backward-compatible support for older naming
    if not ite_cols:
        ite_cols = [c for c in ite_df.columns if c.startswith("ITE_Temp_")]

    if not ite_cols:
        raise ValueError(
            "No Temp ITE columns were found. Please check whether the input file is the Temp ITE output."
        )

    ite_df = ite_df.copy()

    if "ITE_mean" in ite_df.columns:
        ite_df["ITE_Temp_mean"] = ite_df["ITE_mean"]
    else:
        ite_df["ITE_Temp_mean"] = ite_df[ite_cols].mean(axis=1)

    return ite_df


# ============================================================
# Convert standardized-space ITE into raw-unit dY/dT (per °C)
# Formula: theta_raw = theta_z * (sd_Y / sd_T)
# ============================================================
def ite_z_to_dydt_raw(df_obs: pd.DataFrame, ite_mean: np.ndarray) -> np.ndarray:
    sd_T = df_obs[TREAT].std(ddof=1)
    sd_Y = df_obs[OUTCOME].std(ddof=1)

    if sd_T == 0 or sd_Y == 0:
        raise ValueError("sd_T or sd_Y is 0. Raw-unit restoration cannot be performed.")

    dydt = ite_mean * (sd_Y / sd_T)
    return dydt


# ============================================================
# Read cleaned Wang temperature file
# Required columns: Year, RCP, Temp_MayOct_mean
# ============================================================
def load_clean_wang_temperature(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    required_cols = ["Year", "RCP", "Temp_MayOct_mean"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The cleaned Wang temperature file is missing required columns: {missing_cols}"
        )

    df = df[required_cols].copy()
    df["Year"] = df["Year"].astype(str).str.strip()
    df["RCP"] = df["RCP"].astype(str).str.strip()
    df["Temp_MayOct_mean"] = pd.to_numeric(df["Temp_MayOct_mean"], errors="coerce")

    df = df.dropna(subset=["Year", "RCP", "Temp_MayOct_mean"]).reset_index(drop=True)
    return df


def find_baseline_temp(temp_mean: pd.DataFrame, baseline_year_label: str) -> float:
    temp_mean = temp_mean.copy()
    base_rows = temp_mean[temp_mean["Year"] == baseline_year_label].copy()

    if base_rows.empty:
        raise ValueError(
            f"Baseline Year == '{baseline_year_label}' was not found in the cleaned Wang temperature file."
        )

    return float(base_rows["Temp_MayOct_mean"].mean())


# ============================================================
# Build the local slope curve dY/dT(T) in raw units
# ============================================================
def build_local_slope_curve(
    df_obs: pd.DataFrame,
    ite_df: pd.DataFrame,
    bin_width: float = 1.0,
    robust: str = "mean",
) -> pd.DataFrame:
    sd_T = df_obs["Temp"].std(ddof=1)
    sd_Y = df_obs["Isoprene"].std(ddof=1)

    if "ITE_Temp_mean" not in ite_df.columns:
        ite_cols = [c for c in ite_df.columns if c.startswith("ITE_") and c != "ITE_mean"]
        if not ite_cols:
            ite_cols = [c for c in ite_df.columns if c.startswith("ITE_Temp_")]
        ite_df = ite_df.copy()
        ite_df["ITE_Temp_mean"] = ite_df[ite_cols].mean(axis=1)

    dydt_raw = ite_df["ITE_Temp_mean"].values * (sd_Y / sd_T)

    df_tmp = df_obs[["Temp", "Month"]].copy()
    df_tmp["dYdT_perC_raw"] = dydt_raw

    df_tmp["Temp_bin"] = np.floor(df_tmp["Temp"] / bin_width) * bin_width

    aggfunc = np.mean if robust == "mean" else np.median

    slope_curve = (
        df_tmp.groupby("Temp_bin", as_index=False)["dYdT_perC_raw"]
        .agg(dYdT=aggfunc, n_samples="count")
        .sort_values("Temp_bin")
        .reset_index(drop=True)
    )

    return slope_curve


def integrate_delta_isoprene(T0: float, T1: float, slope_curve: pd.DataFrame) -> float:
    """
    Compute:
        Delta_Isoprene = integral from T0 to T1 of f(T) dT

    using linear interpolation of the local slope curve and trapezoidal integration.
    """
    if T0 == T1:
        return 0.0

    sign = 1.0
    if T1 < T0:
        T0, T1 = T1, T0
        sign = -1.0

    x = slope_curve["Temp_bin"].values.astype(float)
    y = slope_curve["dYdT"].values.astype(float)

    def f(temp):
        temp = np.clip(temp, x.min(), x.max())
        return np.interp(temp, x, y)

    step = 0.25
    grid = np.arange(T0, T1 + 1e-9, step)
    vals = np.array([f(t) for t in grid])

    delta = np.trapz(vals, grid)
    return sign * float(delta)


def project_isoprene_change_local_ite(
    temp_mean: pd.DataFrame,
    T_base: float,
    Y_base: float,
    slope_curve: pd.DataFrame,
) -> pd.DataFrame:
    out = temp_mean.copy()
    out["T_base"] = T_base
    out["Isoprene_base"] = Y_base

    deltas = []
    for Tf in out["Temp_MayOct_mean"].values:
        deltas.append(integrate_delta_isoprene(T_base, float(Tf), slope_curve))

    out["Delta_T_vs_base"] = out["Temp_MayOct_mean"] - T_base
    out["Delta_Isoprene_vs_base"] = deltas
    out["Isoprene_future"] = out["Isoprene_base"] + out["Delta_Isoprene_vs_base"]
    out["Isoprene_pct_change (%)"] = (
        out["Delta_Isoprene_vs_base"] / out["Isoprene_base"]
    ) * 100

    return out


# ============================================================
# Main workflow
# ============================================================
def main():
    # (A) Read observational data
    df_obs = load_obs_data(DATA_PATH)

    # (B) Read Temp ITE and convert to raw dY/dT
    ite_df = load_temp_ite(ITE_TEMP_CSV, nrows_expected=len(df_obs))
    dydt = ite_z_to_dydt_raw(df_obs, ite_df["ITE_Temp_mean"].values)

    theta_mean = float(np.mean(dydt))
    theta_median = float(np.median(dydt))

    rowwise = df_obs[[TREAT, OUTCOME, "Month", "Hour"]].copy()
    rowwise["ITE_Temp_mean_z"] = ite_df["ITE_Temp_mean"].values
    rowwise["dYdT_perC_raw"] = dydt
    rowwise.to_csv(os.path.join(OUT_DIR, "Temp_ITE_rowwise_dYdT.csv"), index=False)

    # (C) Read cleaned Wang temperature file
    temp_mean = load_clean_wang_temperature(WANG_TEMP_CLEAN_FILE)
    temp_mean.to_csv(os.path.join(OUT_DIR, "wang_yearly_temp_mean_MayOct_used.csv"), index=False)

    # (D) Baseline temperature from the cleaned Wang file
    T_wang_base = find_baseline_temp(temp_mean, baseline_year_label=BASELINE_YEAR_LABEL)

    # (E) Build local ITE slope curve
    slope_curve = build_local_slope_curve(
        df_obs=df_obs,
        ite_df=ite_df,
        bin_width=1.0,
        robust="mean",
    )
    slope_curve.to_csv(os.path.join(OUT_DIR, "Temp_local_slope_curve_1C.csv"), index=False)

    # (F) Use observational means as the corrected baseline
    mu_T = df_obs[TREAT].mean()
    mu_Y = df_obs[OUTCOME].mean()

    T_base = mu_T
    Y_base = mu_Y

    temp_mean = temp_mean.copy()
    temp_mean["Temp_corrected"] = mu_T + (temp_mean["Temp_MayOct_mean"] - T_wang_base)

    temp_mean_corrected = temp_mean.drop(columns=["Temp_MayOct_mean"]).rename(
        columns={"Temp_corrected": "Temp_MayOct_mean"}
    )

    temp_mean_corrected = temp_mean_corrected[
        temp_mean_corrected["Year"] != BASELINE_YEAR_LABEL
    ].copy()

    baseline_rows = pd.DataFrame({
        "Year": ["2020", "2020", "2020"],
        "RCP": ["RCP4.5", "RCP6.0", "RCP8.5"],
        "Temp_MayOct_mean": [mu_T, mu_T, mu_T],
        "T_base": [mu_T, mu_T, mu_T],
        "Isoprene_base": [mu_Y, mu_Y, mu_Y],
        "Delta_T_vs_base": [0.0, 0.0, 0.0],
        "Delta_Isoprene_vs_base": [0.0, 0.0, 0.0],
        "Isoprene_future": [mu_Y, mu_Y, mu_Y],
        "Isoprene_pct_change (%)": [0.0, 0.0, 0.0],
    })

    # (G) Project future isoprene using local ITE integration
    proj_local = project_isoprene_change_local_ite(
        temp_mean=temp_mean_corrected,
        T_base=T_base,
        Y_base=Y_base,
        slope_curve=slope_curve,
    )

    proj_local = pd.concat([baseline_rows, proj_local], ignore_index=True)

    out_path = os.path.join(OUT_DIR, "future_isoprene_projection_yearly_localITE.csv")
    proj_local.to_csv(out_path, index=False)

    # (H) Save summary text
    summary_path = os.path.join(OUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Temperature projection based on local ITE ===\n")
        f.write(f"Baseline temperature (observed mean Temp): {T_base}\n")
        f.write(f"Baseline isoprene (observed mean Isoprene): {Y_base}\n")
        f.write(f"Wang baseline temperature ({BASELINE_YEAR_LABEL}, May-Oct mean): {T_wang_base}\n")
        f.write(f"Mean dY/dT: {theta_mean}\n")
        f.write(f"Median dY/dT: {theta_median}\n")
        f.write(f"Projection output: {out_path}\n")

    print("\n=== DONE ===")
    print("Rowwise dY/dT saved:", os.path.join(OUT_DIR, "Temp_ITE_rowwise_dYdT.csv"))
    print("Cleaned Wang temperature used:", os.path.join(OUT_DIR, "wang_yearly_temp_mean_MayOct_used.csv"))
    print("Local slope curve saved:", os.path.join(OUT_DIR, "Temp_local_slope_curve_1C.csv"))
    print("Local-ITE future projection saved:", out_path)
    print("Summary saved:", summary_path)


if __name__ == "__main__":
    main()
