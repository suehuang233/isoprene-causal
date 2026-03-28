import os
import numpy as np
import pandas as pd

# =========================
# User configuration
# =========================
DATA_PATH = "./data/month5to10-5to20hour-2020.csv"
ITE_TEMP_CSV = "./isoprene_dowhy_causaleffect/Temp_ITE.csv"
OUT_DIR = "./isoprene_dowhy_causaleffect/postprocess_temp_curve"
os.makedirs(OUT_DIR, exist_ok=True)

# Analysis window (must match the modeling setup)
MONTH_MIN, MONTH_MAX = 5, 10
HOUR_MIN, HOUR_MAX = 5, 20

OUTCOME = "Isoprene"
TREAT = "Temp"

# =========================
# Read and align data
# =========================
df = pd.read_csv(DATA_PATH)
df = df[(df["Month"] >= MONTH_MIN) & (df["Month"] <= MONTH_MAX)]
df = df[(df["Hour"] >= HOUR_MIN) & (df["Hour"] <= HOUR_MAX)]
df = df.reset_index(drop=True)

ite_df = pd.read_csv(ITE_TEMP_CSV)

# The ITE file is expected to be created in the main pipeline using the same row order.
if len(df) != len(ite_df):
    raise ValueError(
        f"Row mismatch: df={len(df)}, ite_df={len(ite_df)}. "
        "Please make sure the ITE file corresponds to the same filtered window "
        "and the same data version."
    )

# Find ITE columns
# Compatible with both naming styles:
# - ITE_Temp_0, ITE_Temp_1, ...
# - ITE_1, ITE_2, ... (if the file is Temp-specific)
ite_cols = [c for c in ite_df.columns if c.startswith("ITE_Temp_")]
if not ite_cols:
    ite_cols = [c for c in ite_df.columns if c.startswith("ITE_")]

if not ite_cols:
    raise ValueError(
        "No ITE columns were found. Please check whether ITE_TEMP_CSV is the Temp output file."
    )

# Average across iterations
ite_df["ITE_Temp_mean"] = ite_df[ite_cols].mean(axis=1)

# =========================
# Compute mu/sigma for inverse transformation
# These must be based on the same filtered dataset used in modeling
# =========================
mu_T = df[TREAT].mean()
sd_T = df[TREAT].std(ddof=1)

mu_Y = df[OUTCOME].mean()
sd_Y = df[OUTCOME].std(ddof=1)

if sd_T == 0 or sd_Y == 0:
    raise ValueError("sd_T or sd_Y is 0. Unit restoration cannot be performed.")

# =========================
# Convert standardized-space ITE to raw-unit slope dY/dT (per °C)
# theta_raw = theta_z * (sd_Y / sd_T)
# =========================
theta_raw = ite_df["ITE_Temp_mean"].values * (sd_Y / sd_T)

# =========================
# Utility function:
# given any raw temperature series, predict raw isoprene
# =========================
def predict_isoprene_from_temp(temp_raw_array):
    """
    Parameters
    ----------
    temp_raw_array : array-like or scalar
        Temperature values in raw units.

    Returns
    -------
    array-like
        Predicted isoprene in raw concentration units.

    Interpretation
    --------------
    The prediction is anchored at (mu_T, mu_Y), while preserving the
    sample-specific slope structure through theta_raw.
    """
    temp_raw_array = np.asarray(temp_raw_array)
    return mu_Y + theta_raw * (temp_raw_array - mu_T)

# =========================
# Row-wise output
# =========================
df_out = df[[TREAT, OUTCOME, "Month", "Hour"]].copy()
df_out["ITE_Temp_mean"] = ite_df["ITE_Temp_mean"].values

# Standardized variables
df_out["Temp_z"] = (df_out[TREAT] - mu_T) / sd_T
df_out["Isoprene_z"] = (df_out[OUTCOME] - mu_Y) / sd_Y

# Standardized-space prediction
df_out["Yhat_z"] = df_out["ITE_Temp_mean"] * df_out["Temp_z"]

# Raw-unit slope and raw-unit prediction
df_out["dYdT_perC"] = df_out["ITE_Temp_mean"] * (sd_Y / sd_T)
df_out["Yhat_raw"] = mu_Y + df_out["dYdT_perC"] * (df_out[TREAT] - mu_T)

rowwise_path = os.path.join(OUT_DIR, "Temp_ITE_to_raw_isoprene_rowwise.csv")
df_out.to_csv(rowwise_path, index=False)

# =========================
# Binned curve by raw temperature
# =========================
bin_width_raw = 1.0
df_out["Temp_bin_raw"] = np.floor(df_out[TREAT] / bin_width_raw) * bin_width_raw

curve = (
    df_out.groupby("Temp_bin_raw", as_index=False)
    .agg(
        Temp_raw_mean=(TREAT, "mean"),
        Temp_z_mean=("Temp_z", "mean"),

        Y_obs_raw_mean=(OUTCOME, "mean"),
        Y_obs_z_mean=("Isoprene_z", "mean"),

        Y_hat_raw_mean=("Yhat_raw", "mean"),
        Y_hat_raw_std=("Yhat_raw", "std"),
        Y_hat_z_mean=("Yhat_z", "mean"),
        Y_hat_z_std=("Yhat_z", "std"),

        dYdT_perC_mean=("dYdT_perC", "mean"),
        ITE_z_mean=("ITE_Temp_mean", "mean"),
        n=("Yhat_raw", "size"),
    )
    .sort_values("Temp_bin_raw")
)

curve_path = os.path.join(OUT_DIR, "Temp_curve_binned_1C.csv")
curve.to_csv(curve_path, index=False)

print("Saved:")
print(" - rowwise:", rowwise_path)
print(" - binned curve:", curve_path)

# =========================
# Optional example for future warming scenarios
# Uncomment if needed
# =========================
# temp_future = df_out[TREAT].values + 2.0
# df_out["Yhat_future_plus2C"] = predict_isoprene_from_temp(temp_future)
# future_path = os.path.join(OUT_DIR, "Temp_future_plus2C_isoprene.csv")
# df_out.to_csv(future_path, index=False)
# print(" - future (+2C example):", future_path)
