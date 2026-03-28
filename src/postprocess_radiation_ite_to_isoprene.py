import os
import numpy as np
import pandas as pd

# =========================
# User configuration
# =========================
DATA_PATH = "./data/month5to10-5to20hour-2020.csv"
ITE_RAD_CSV = "./isoprene_dowhy_causaleffect/Radiation_ITE.csv"
OUT_DIR = "./isoprene_dowhy_causaleffect/postprocess_radiation_curve"
os.makedirs(OUT_DIR, exist_ok=True)

# Analysis window (must match the modeling setup)
MONTH_MIN, MONTH_MAX = 5, 10
HOUR_MIN, HOUR_MAX = 5, 20

OUTCOME = "Isoprene"
TREAT = "Radiation"

# =========================
# Read and align data
# =========================
df = pd.read_csv(DATA_PATH)
df = df[(df["Month"] >= MONTH_MIN) & (df["Month"] <= MONTH_MAX)]
df = df[(df["Hour"] >= HOUR_MIN) & (df["Hour"] <= HOUR_MAX)]
df = df.reset_index(drop=True)

ite_df = pd.read_csv(ITE_RAD_CSV)

if len(df) != len(ite_df):
    raise ValueError(
        f"Row mismatch: df={len(df)}, ite_df={len(ite_df)}. "
        "Please make sure the ITE_Radiation file corresponds to the same filtered window "
        "and the same data version."
    )

# Find ITE columns
# Compatible with both naming styles:
# - ITE_Radiation_0, ITE_Radiation_1, ...
# - ITE_1, ITE_2, ... (if the file is Radiation-specific)
ite_cols = [c for c in ite_df.columns if c.startswith("ITE_Radiation_")]
if not ite_cols:
    ite_cols = [c for c in ite_df.columns if c.startswith("ITE_")]

if not ite_cols:
    raise ValueError(
        "No ITE_Radiation columns were found. Please check whether ITE_RAD_CSV is the Radiation output file."
    )

# Average across iterations
ite_df["ITE_Radiation_mean"] = ite_df[ite_cols].mean(axis=1)

# =========================
# Compute mu/sigma for inverse transformation
# =========================
mu_T = df[TREAT].mean()
sd_T = df[TREAT].std(ddof=1)

mu_Y = df[OUTCOME].mean()
sd_Y = df[OUTCOME].std(ddof=1)

if sd_T == 0 or sd_Y == 0:
    raise ValueError("sd_T or sd_Y is 0. Unit restoration cannot be performed.")

# =========================
# Convert standardized-space ITE to raw-unit slope dY/dRadiation
# theta_raw = theta_z * (sd_Y / sd_T)
# =========================
theta_raw = ite_df["ITE_Radiation_mean"].values * (sd_Y / sd_T)

# =========================
# Row-wise output
# =========================
df_out = df[[TREAT, OUTCOME, "Month", "Hour"]].copy()
df_out["ITE_Radiation_mean"] = ite_df["ITE_Radiation_mean"].values

# Standardized variables
df_out["Radiation_z"] = (df_out[TREAT] - mu_T) / sd_T
df_out["Isoprene_z"] = (df_out[OUTCOME] - mu_Y) / sd_Y

# Standardized-space prediction
df_out["Yhat_z"] = df_out["ITE_Radiation_mean"] * df_out["Radiation_z"]

# Raw-unit slope and raw-unit prediction
df_out["dYdRad_perUnit"] = df_out["ITE_Radiation_mean"] * (sd_Y / sd_T)
df_out["Yhat_raw"] = mu_Y + df_out["dYdRad_perUnit"] * (df_out[TREAT] - mu_T)

rowwise_path = os.path.join(OUT_DIR, "Radiation_ITE_to_raw_isoprene_rowwise.csv")
df_out.to_csv(rowwise_path, index=False)

# =========================
# Binned curve by raw Radiation
# Adjust bin_width_raw if needed based on the physical unit scale
# =========================
bin_width_raw = 50.0
df_out["Radiation_bin_raw"] = np.floor(df_out[TREAT] / bin_width_raw) * bin_width_raw

curve = (
    df_out.groupby("Radiation_bin_raw", as_index=False)
    .agg(
        Radiation_raw_mean=(TREAT, "mean"),
        Radiation_z_mean=("Radiation_z", "mean"),

        Y_obs_raw_mean=(OUTCOME, "mean"),
        Y_obs_z_mean=("Isoprene_z", "mean"),

        Y_hat_raw_mean=("Yhat_raw", "mean"),
        Y_hat_raw_std=("Yhat_raw", "std"),
        Y_hat_z_mean=("Yhat_z", "mean"),
        Y_hat_z_std=("Yhat_z", "std"),

        dYdRad_perUnit_mean=("dYdRad_perUnit", "mean"),
        ITE_z_mean=("ITE_Radiation_mean", "mean"),
        n=("Yhat_raw", "size"),
    )
    .sort_values("Radiation_bin_raw")
)

curve_path = os.path.join(OUT_DIR, "Radiation_curve_binned.csv")
curve.to_csv(curve_path, index=False)

print("Saved:")
print(" - rowwise:", rowwise_path)
print(" - binned curve:", curve_path)
