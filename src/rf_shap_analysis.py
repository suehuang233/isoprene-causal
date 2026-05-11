#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random Forest and SHAP analysis for isoprene.

This script trains a Random Forest regression model, evaluates predictive
performance, computes SHAP values, permutation importance, and bootstrap-based
SHAP stability statistics.
"""

import os
import warnings

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split


warnings.filterwarnings("ignore")


# =============================================================================
# User configuration
# =============================================================================
SEED = 2026
np.random.seed(SEED)

# Place the input CSV in the same directory as this script, or replace this path.
DATA_PATH = "../data/month5to10-5to20hour-2020.csv"

OUTPUT_DIR = "../results/shap_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MONTH_MIN, MONTH_MAX = 5, 10
HOUR_MIN, HOUR_MAX = 5, 20

FEATURES = [
    "RH", "Temp", "Radiation", "u", "v",
    "Oxides", "Toluene", "OH",
]

TARGET = "Isoprene"

TEST_SIZE = 0.2
N_RANDOM_SEARCH_ITER = 40
N_BOOTSTRAP = 30
BOOTSTRAP_SAMPLE_RATIO = 0.8
PERMUTATION_REPEATS = 20


# =============================================================================
# Load data
# =============================================================================
df = pd.read_csv(DATA_PATH)

df = df[(df["Month"] >= MONTH_MIN) & (df["Month"] <= MONTH_MAX)]
df = df[(df["Hour"] >= HOUR_MIN) & (df["Hour"] <= HOUR_MAX)]

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

missing_features = [feature for feature in FEATURES if feature not in df.columns]
if missing_features:
    raise ValueError(f"Missing required feature columns: {missing_features}")
if TARGET not in df.columns:
    raise ValueError(f"Missing target column: {TARGET}")

X = df[FEATURES].copy()
y = df[TARGET].copy()

print("=" * 80)
print("Data loaded")
print(f"Samples: {len(df)}")
print(f"Features: {FEATURES}")
print("=" * 80)


# =============================================================================
# Train/test split
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=SEED,
)

print(f"Train size: {len(X_train)}")
print(f"Test size : {len(X_test)}")


# =============================================================================
# Hyperparameter tuning
# =============================================================================
print("=" * 80)
print("RandomizedSearchCV for Random Forest")
print("=" * 80)

base_model = RandomForestRegressor(
    random_state=SEED,
    n_jobs=-1,
    bootstrap=True,
    oob_score=True,
)

param_dist = {
    "n_estimators": [200, 300, 500, 800],
    "max_depth": [4, 6, 8],
    "min_samples_split": [10, 20, 50],
    "min_samples_leaf": [5, 10, 20],
    "max_features": [0.3, 0.5, "sqrt"],
    "max_samples": [0.6, 0.8],
    "ccp_alpha": [0.0, 1e-4, 1e-3],
}

cv_inner = KFold(n_splits=5, shuffle=True, random_state=SEED)

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=N_RANDOM_SEARCH_ITER,
    scoring="r2",
    cv=cv_inner,
    verbose=1,
    random_state=SEED,
    n_jobs=-1,
    refit=True,
)

random_search.fit(X_train, y_train)

model = random_search.best_estimator_
best_params = random_search.best_params_
best_cv_score = random_search.best_score_

print("Best parameters:")
print(best_params)
print(f"Best CV R2: {best_cv_score:.4f}")

pd.DataFrame([best_params]).to_csv(
    os.path.join(OUTPUT_DIR, "best_hyperparameters.csv"),
    index=False,
)

pd.DataFrame([{"best_cv_r2": best_cv_score}]).to_csv(
    os.path.join(OUTPUT_DIR, "best_cv_score.csv"),
    index=False,
)


# =============================================================================
# Cross-validation on the training set
# =============================================================================
cv_outer = KFold(n_splits=5, shuffle=True, random_state=SEED)

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv_outer,
    scoring={
        "r2": "r2",
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
    },
    n_jobs=-1,
    return_train_score=True,
)

cv_summary = pd.DataFrame(
    [
        {
            "cv_train_r2_mean": np.mean(cv_results["train_r2"]),
            "cv_train_r2_std": np.std(cv_results["train_r2"]),
            "cv_valid_r2_mean": np.mean(cv_results["test_r2"]),
            "cv_valid_r2_std": np.std(cv_results["test_r2"]),
            "cv_valid_rmse_mean": -np.mean(cv_results["test_neg_rmse"]),
            "cv_valid_rmse_std": np.std(-cv_results["test_neg_rmse"]),
            "cv_valid_mae_mean": -np.mean(cv_results["test_neg_mae"]),
            "cv_valid_mae_std": np.std(-cv_results["test_neg_mae"]),
        }
    ]
)

cv_summary.to_csv(os.path.join(OUTPUT_DIR, "cv_summary.csv"), index=False)
print(cv_summary)


# =============================================================================
# Final fit and model metrics
# =============================================================================
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

metrics_df = pd.DataFrame(
    [
        {
            "r2_train": r2_score(y_train, y_pred_train),
            "r2_test": r2_score(y_test, y_pred_test),
            "rmse_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "mae_train": mean_absolute_error(y_train, y_pred_train),
            "mae_test": mean_absolute_error(y_test, y_pred_test),
            "r2_gap": r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test),
            "oob_score": getattr(model, "oob_score_", np.nan),
        }
    ]
)

metrics_df.to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"), index=False)
print(metrics_df)

pred_df = pd.DataFrame(
    {
        "y_true_test": y_test.values,
        "y_pred_test": y_pred_test,
        "residual": y_test.values - y_pred_test,
    }
)
pred_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)


# =============================================================================
# Random Forest feature importance
# =============================================================================
rf_importance = pd.DataFrame(
    {
        "feature": FEATURES,
        "rf_feature_importance": model.feature_importances_,
    }
).sort_values("rf_feature_importance", ascending=False)

rf_importance.to_csv(os.path.join(OUTPUT_DIR, "rf_feature_importance.csv"), index=False)
print(rf_importance)


# =============================================================================
# SHAP on the test set
# =============================================================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

mean_abs_shap = np.abs(shap_values).mean(axis=0)
mean_shap_signed = shap_values.mean(axis=0)

importance_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap_signed": mean_shap_signed,
    }
).sort_values("mean_abs_shap", ascending=False)

importance_df.to_csv(os.path.join(OUTPUT_DIR, "shap_importance.csv"), index=False)
print(importance_df)

shap_values_df = pd.DataFrame(shap_values, columns=FEATURES)
shap_values_df.to_csv(os.path.join(OUTPUT_DIR, "shap_values_test.csv"), index=False)

base_value_main = explainer.expected_value
if isinstance(base_value_main, (list, np.ndarray)):
    base_value_main = np.asarray(base_value_main).reshape(-1)[0]
base_value_main = float(base_value_main)


# =============================================================================
# Permutation importance
# =============================================================================
perm = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=PERMUTATION_REPEATS,
    random_state=SEED,
    n_jobs=-1,
)

perm_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    }
).sort_values("perm_importance_mean", ascending=False)

perm_df.to_csv(os.path.join(OUTPUT_DIR, "permutation_importance.csv"), index=False)


# =============================================================================
# SHAP dependence data
# =============================================================================
for var in ["Temp", "Radiation"]:
    if var in FEATURES:
        idx = FEATURES.index(var)
        dep_df = pd.DataFrame(
            {
                var: X_test[var].values,
                "shap_value": shap_values[:, idx],
            }
        )
        dep_df.to_csv(
            os.path.join(OUTPUT_DIR, f"shap_dependence_{var}.csv"),
            index=False,
        )

print("SHAP done.")


# =============================================================================
# Bootstrap stability analysis: retrain the model each time
# =============================================================================
robust_results = []

for i in range(N_BOOTSTRAP):
    n_samples = int(len(X_train) * BOOTSTRAP_SAMPLE_RATIO)
    idx = np.random.choice(len(X_train), size=n_samples, replace=True)

    X_boot = X_train.iloc[idx].reset_index(drop=True)
    y_boot = y_train.iloc[idx].reset_index(drop=True)

    model_boot = RandomForestRegressor(
        **best_params,
        random_state=SEED + i,
        n_jobs=-1,
        bootstrap=True,
        oob_score=False,
    )

    model_boot.fit(X_boot, y_boot)

    explainer_boot = shap.TreeExplainer(model_boot)
    shap_boot = explainer_boot.shap_values(X_test)

    if isinstance(shap_boot, list):
        shap_boot = shap_boot[0]

    mean_abs_shap_boot = np.abs(shap_boot).mean(axis=0)
    mean_shap_signed_boot = shap_boot.mean(axis=0)

    rank_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "mean_abs_shap": mean_abs_shap_boot,
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    rank_df["rank"] = np.arange(1, len(FEATURES) + 1)

    for j, var in enumerate(FEATURES):
        var_rank = rank_df.loc[rank_df["feature"] == var, "rank"].values[0]
        robust_results.append(
            {
                "bootstrap_id": i,
                "feature": var,
                "mean_abs_shap": mean_abs_shap_boot[j],
                "mean_shap_signed": mean_shap_signed_boot[j],
                "rank": var_rank,
            }
        )

robust_df = pd.DataFrame(robust_results)

stability_df = robust_df.groupby("feature").agg(
    mean_importance=("mean_abs_shap", "mean"),
    std_importance=("mean_abs_shap", "std"),
    mean_rank=("rank", "mean"),
    std_rank=("rank", "std"),
    min_rank=("rank", "min"),
    max_rank=("rank", "max"),
).reset_index()

stability_df["cv"] = stability_df["std_importance"] / stability_df["mean_importance"]
stability_df = stability_df.sort_values("mean_importance", ascending=False)

stability_df.to_csv(os.path.join(OUTPUT_DIR, "shap_stability_stats.csv"), index=False)
robust_df.to_csv(os.path.join(OUTPUT_DIR, "shap_robustness.csv"), index=False)

print("Saved revised SHAP curves with std.")
print("Bootstrap stability analysis finished.")


# =============================================================================
# Plot 1: SHAP bar plot with beeswarm overlay
# =============================================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

feature_names = list(X_test.columns)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
sort_idx = np.argsort(mean_abs_shap)[::-1]

feature_names = [feature_names[i] for i in sort_idx]
display_names = [
    "Temperature" if feature == "Temp" else
    "Anth. proxy" if feature == "Toluene" else
    feature
    for feature in feature_names
]

mean_abs_shap = mean_abs_shap[sort_idx]
shap_values_sorted = shap_values[:, sort_idx]
X_sorted = X_test.iloc[:, sort_idx].copy()

n_features = len(feature_names)
y_pos = np.arange(n_features)

cmap = shap.plots.colors.red_blue

normed_X = X_sorted.copy()
for col in normed_X.columns:
    x = normed_X[col].values.astype(float)
    xmin, xmax = np.nanpercentile(x, 5), np.nanpercentile(x, 95)
    if xmax == xmin:
        normed_X[col] = 0.5
    else:
        x = np.clip(x, xmin, xmax)
        normed_X[col] = (x - xmin) / (xmax - xmin)

fig = plt.figure(figsize=(9.2, 6.4), dpi=600)
ax = fig.add_axes([0.12, 0.12, 0.72, 0.78])
ax_top = ax.twiny()

ax.set_zorder(2)
ax_top.set_zorder(1)

ax.patch.set_alpha(0)
ax_top.patch.set_alpha(0)

bar_colors = ["#d9d9d9"] * n_features

ax_top.barh(
    y_pos,
    mean_abs_shap,
    height=0.72,
    color=bar_colors,
    edgecolor="none",
    alpha=0.88,
    zorder=2,
)

ax_top.set_ylim(-0.6, n_features - 0.4)
ax_top.invert_yaxis()
ax_top.set_xlim(0, max(mean_abs_shap) * 1.15)
ax_top.set_xlabel("Mean |SHAP value|", fontsize=22, fontweight="bold")
ax_top.xaxis.set_label_position("top")
ax_top.xaxis.tick_top()
ax_top.tick_params(axis="x", labelsize=20, width=1.0, length=5)
ax_top.tick_params(axis="y", left=False, labelleft=False)
ax_top.spines["right"].set_visible(False)
ax_top.spines["bottom"].set_visible(False)
ax_top.spines["left"].set_visible(False)
ax_top.spines["top"].set_linewidth(2)

rng = np.random.default_rng(SEED)

for i, col in enumerate(feature_names):
    shap_col = shap_values_sorted[:, i]
    val_col = normed_X[col].values

    order = np.argsort(shap_col)
    shap_col = shap_col[order]
    val_col = val_col[order]

    jitter = rng.normal(0, 0.08, size=len(shap_col))
    y_plot = np.full(len(shap_col), i) + jitter

    ax.scatter(
        shap_col,
        y_plot,
        c=val_col,
        cmap=cmap,
        s=14,
        alpha=0.95,
        linewidths=0,
        zorder=5,
    )

ax.set_ylim(-0.6, n_features - 0.4)
ax.invert_yaxis()
ax.set_yticks(y_pos)
ax.set_yticklabels(display_names, fontsize=18, fontweight="bold")
ax.set_xlabel("SHAP value (impact on model output)", fontsize=22, fontweight="bold")
ax.set_ylabel("Features", fontsize=22, fontweight="bold")
ax.tick_params(axis="x", labelsize=20, width=1.0, length=5)
ax.tick_params(axis="y", width=0, length=0)

ax.axvline(0, color="gray", lw=1.6, alpha=0.8, zorder=0, linestyle="--")
ax.grid(axis="y", linestyle=(0, (1, 4)), color="#c9c9c9", linewidth=0.7)
ax.set_axisbelow(True)

ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(True)
ax.spines["right"].set_linewidth(2)

ax.tick_params(axis="y", right=False, labelright=False)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.8%", pad=0.50)

sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
sm.set_array([])
cb = plt.colorbar(sm, cax=cax)
cb.set_ticks([0, 1])
cb.set_ticklabels(["Low", "High"])
cb.ax.tick_params(labelsize=18, length=0)
cb.set_label("Feature value", fontsize=20, fontweight="bold", rotation=90, labelpad=5)
cb.outline.set_linewidth(1.0)

plt.savefig(os.path.join(OUTPUT_DIR, "shap_combined_template_style_v2.tiff"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "shap_combined_template_style_v2.pdf"), bbox_inches="tight")
plt.show()


# =============================================================================
# Plot 2: signed SHAP bar plot
# =============================================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

mean_signed_shap = shap_values.mean(axis=0)

signed_df = pd.DataFrame(
    {
        "feature": X_test.columns,
        "mean_signed_shap": mean_signed_shap,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }
).sort_values("mean_signed_shap", ascending=True)

signed_df["feature"] = signed_df["feature"].replace(
    {
        "Temp": "Temperature",
        "Toluene": "Anth. proxy",
    }
)

fig, ax = plt.subplots(figsize=(9.2, 6.4), dpi=600)

cmap = shap.plots.colors.red_blue
neg_color = cmap(0.1)
pos_color = cmap(0.85)

colors = [neg_color if x < 0 else pos_color for x in signed_df["mean_signed_shap"]]

ax.barh(
    signed_df["feature"],
    signed_df["mean_signed_shap"],
    color=colors,
    alpha=0.8,
)

ax.axvline(0, color="black", lw=1, linestyle="--")
ax.set_xlabel("Mean signed SHAP value", fontsize=22, fontweight="bold")
ax.set_ylabel("Features", fontsize=22, fontweight="bold")

ax.tick_params(axis="x", labelsize=20)
ax.tick_params(axis="y", labelsize=20)

for label in ax.get_yticklabels():
    label.set_fontweight("bold")

ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)
ax.spines["top"].set_visible(True)
ax.spines["top"].set_linewidth(2)
ax.spines["right"].set_visible(True)
ax.spines["right"].set_linewidth(2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_signed_mean_bar.tiff"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "shap_signed_mean_bar.pdf"), bbox_inches="tight")
plt.show()


# =============================================================================
# Plot 3: predicted vs observed for train and test data
# =============================================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

y_train_true = np.asarray(y_train)
y_train_pred = np.asarray(y_pred_train)
y_test_true = np.asarray(y_test)
y_test_pred = np.asarray(y_pred_test)

r2_train = r2_score(y_train_true, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train_true, y_train_pred))

r2_test = r2_score(y_test_true, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test_true, y_test_pred))

xy_min = min(
    np.min(y_train_true), np.min(y_train_pred),
    np.min(y_test_true), np.min(y_test_pred),
)
xy_max = max(
    np.max(y_train_true), np.max(y_train_pred),
    np.max(y_test_true), np.max(y_test_pred),
)

fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=600)

train_color = "#C03A8C"
test_color = "#9E9E9E"

ax.scatter(
    y_test_true,
    y_test_pred,
    s=40,
    alpha=0.75,
    color=test_color,
    edgecolors="none",
    label="Test data",
    zorder=2,
)

ax.scatter(
    y_train_true,
    y_train_pred,
    s=40,
    alpha=0.55,
    color=train_color,
    edgecolors="none",
    label="Train data",
    zorder=3,
)

ax.plot(
    [xy_min, xy_max],
    [xy_min, xy_max],
    linestyle="--",
    color="gray",
    linewidth=1.6,
    zorder=1,
)

metrics_text = (
    f"Train R² = {r2_train:.3f}, RMSE = {rmse_train:.2f}" + "\n" +
    f"Test R² = {r2_test:.3f}, RMSE = {rmse_test:.2f}"
)

ax.text(
    0.05,
    0.96,
    metrics_text,
    transform=ax.transAxes,
    fontsize=18,
    fontweight="bold",
    fontfamily="Times New Roman",
    ha="left",
    va="top",
    linespacing=1.4,
)

leg = ax.legend(
    loc="lower right",
    frameon=False,
    fontsize=18,
    handletextpad=0.4,
)
for text in leg.get_texts():
    text.set_fontweight("bold")

ax.set_xlabel("Observed Isoprene", fontsize=22, fontweight="bold")
ax.set_ylabel("Predicted Isoprene", fontsize=22, fontweight="bold")

ax.tick_params(axis="both", labelsize=20, width=1.0, length=5)

for label in ax.get_xticklabels():
    label.set_fontweight("bold")
for label in ax.get_yticklabels():
    label.set_fontweight("bold")

ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)
ax.spines["top"].set_linewidth(1.2)
ax.spines["right"].set_linewidth(1.2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "predicted_vs_observed_train_test.tiff"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "predicted_vs_observed_train_test.pdf"), bbox_inches="tight")
plt.show()


# =============================================================================
# Plot 4: residual plot for train and test data
# =============================================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

y_train_true = np.asarray(y_train)
y_train_pred = np.asarray(y_pred_train)
y_test_true = np.asarray(y_test)
y_test_pred = np.asarray(y_pred_test)

res_train = y_train_true - y_train_pred
res_test = y_test_true - y_test_pred

train_color = "#C51B7D"
test_color = "#9E9E9E"

fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=600)

test_scatter = ax.scatter(
    y_test_pred,
    res_test,
    s=28,
    alpha=0.75,
    color=test_color,
    edgecolors="none",
    label="Test",
    zorder=2,
)

train_scatter = ax.scatter(
    y_train_pred,
    res_train,
    s=28,
    alpha=0.55,
    color=train_color,
    edgecolors="none",
    label="Train",
    zorder=3,
)

ax.axhline(
    0,
    linestyle="--",
    color="gray",
    linewidth=1.6,
    zorder=1,
)

leg = ax.legend(
    handles=[train_scatter, test_scatter],
    labels=["Train", "Test"],
    loc="upper left",
    frameon=False,
    fontsize=17,
    handletextpad=0.4,
)
for text in leg.get_texts():
    text.set_fontweight("bold")

ax.set_xlabel("Predicted Isoprene", fontsize=22, fontweight="bold")
ax.set_ylabel("Residuals", fontsize=22, fontweight="bold")

ax.tick_params(axis="both", labelsize=20, width=1.0, length=5)

for label in ax.get_xticklabels():
    label.set_fontweight("bold")
for label in ax.get_yticklabels():
    label.set_fontweight("bold")

ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)
ax.spines["top"].set_linewidth(1.2)
ax.spines["right"].set_linewidth(1.2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residual_plot_train_test.tiff"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "residual_plot_train_test.pdf"), bbox_inches="tight")
plt.show()


# =============================================================================
# Plot 5: permutation importance
# =============================================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

plot_df = perm_df.copy().sort_values("perm_importance_mean", ascending=True).reset_index(drop=True)
plot_df["feature"] = plot_df["feature"].replace(
    {
        "Temp": "Temperature",
        "Toluene": "Anth. proxy",
    }
)

fig, ax = plt.subplots(figsize=(7.6, 6.4), dpi=600)

ax.barh(
    plot_df["feature"],
    plot_df["perm_importance_mean"],
    xerr=plot_df["perm_importance_std"],
    color="#d9d9d9",
    edgecolor="none",
    alpha=0.85,
    error_kw=dict(
        ecolor="gray",
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
    ),
)

ax.set_xlabel("Permutation importance", fontsize=22, fontweight="bold")
ax.set_ylabel("Features", fontsize=22, fontweight="bold")

ax.tick_params(axis="x", labelsize=20, width=1.0, length=5)
ax.tick_params(axis="y", labelsize=20, width=0, length=0)

for label in ax.get_xticklabels():
    label.set_fontweight("bold")
for label in ax.get_yticklabels():
    label.set_fontweight("bold")

ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)
ax.spines["top"].set_linewidth(1.2)
ax.spines["right"].set_linewidth(1.2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "permutation_importance.tiff"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "permutation_importance.pdf"), bbox_inches="tight")
plt.show()


# =============================================================================
# Plot 6: bootstrap SHAP stability
# =============================================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

plot_df = stability_df.copy().sort_values("mean_importance", ascending=True).reset_index(drop=True)
plot_df["feature"] = plot_df["feature"].replace(
    {
        "Temp": "Temperature",
        "Toluene": "Anth. proxy",
    }
)

fig, ax = plt.subplots(figsize=(7.6, 6.4), dpi=600)

ax.barh(
    plot_df["feature"],
    plot_df["mean_importance"],
    xerr=plot_df["std_importance"],
    color="#d9d9d9",
    edgecolor="none",
    alpha=0.85,
    error_kw=dict(
        ecolor="gray",
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
    ),
)

ax.set_xlabel("Bootstrap mean |SHAP value|", fontsize=22, fontweight="bold")
ax.set_ylabel("Features", fontsize=22, fontweight="bold")

ax.tick_params(axis="x", labelsize=20, width=1.0, length=5)
ax.tick_params(axis="y", labelsize=20, width=0, length=0)

for label in ax.get_xticklabels():
    label.set_fontweight("bold")
for label in ax.get_yticklabels():
    label.set_fontweight("bold")

ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)
ax.spines["top"].set_linewidth(1.2)
ax.spines["right"].set_linewidth(1.2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "bootstrap_shap_stability.tiff"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "bootstrap_shap_stability.pdf"), bbox_inches="tight")
plt.show()