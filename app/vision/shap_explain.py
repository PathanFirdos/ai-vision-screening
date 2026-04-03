"""
shap_explain.py
---------------
Explains the RandomForest model predictions using SHAP values.

Fixes applied:
  - Multiclass handled correctly (3 plots, one per risk class)
  - Uses updated model bundle format (dict with feature_names)
  - Saves plots to reports/shap_*.png instead of just showing them
  - Added bar chart + beeswarm plot for each class
  - Sample size limited to 500 for speed

Install SHAP if needed:
    pip install shap

Run from project root:
    python ml/shap_explain.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
#  Paths                                                              #
# ------------------------------------------------------------------ #

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "saved_model.pkl")
DATA_PATH   = os.path.join(BASE_DIR, "..", "data",   "vision_dataset.csv")
REPORT_DIR  = os.path.join(BASE_DIR, "..", "reports", "shap")

os.makedirs(REPORT_DIR, exist_ok=True)

# ------------------------------------------------------------------ #
#  Load model + data                                                  #
# ------------------------------------------------------------------ #

print("=" * 50)
print("  SHAP Explainability Report")
print("=" * 50)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}\n"
        "Run ml/train_model.py first."
    )

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}\n"
        "Run ml/generate_dataset.py first."
    )

print("\n[1] Loading model and dataset ...")
bundle        = joblib.load(MODEL_PATH)
model         = bundle["model"]
feature_names = bundle.get("feature_names",
                ["accuracy","response_time","blink_rate",
                 "gaze_stability","distance"])
risk_names    = bundle.get("risk_names", {0:"Low",1:"Moderate",2:"High"})

df = pd.read_csv(DATA_PATH)
X  = df[feature_names]

# Use 500 samples for speed (SHAP can be slow on 10k rows)
X_sample = X.sample(n=min(500, len(X)), random_state=42)
print(f"    Using {len(X_sample)} samples for SHAP analysis.")

# ------------------------------------------------------------------ #
#  Compute SHAP values                                                #
# ------------------------------------------------------------------ #

try:
    import shap
except ImportError:
    raise ImportError("SHAP not installed. Run: pip install shap")

print("\n[2] Computing SHAP values (this may take 30–60 seconds) ...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# shap_values is a list of 3 arrays for multiclass (one per class)
# Shape: [n_classes][n_samples, n_features]
print(f"    SHAP values computed. Classes: {len(shap_values)}")

# ------------------------------------------------------------------ #
#  Plot 1 — Summary bar chart (mean |SHAP|) for ALL classes          #
# ------------------------------------------------------------------ #

print("\n[3] Generating plots ...")

class_labels = [risk_names[i] for i in range(len(risk_names))]

# Bar chart: average impact per feature per class
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("SHAP Feature Importance — Vision Screening Model",
             fontsize=14, fontweight="bold")

for i, (ax, label) in enumerate(zip(axes, class_labels)):
    mean_shap = np.abs(shap_values[i]).mean(axis=0)
    sorted_idx = np.argsort(mean_shap)
    colors = ["#e74c3c" if i == 2 else
              "#f39c12" if i == 1 else
              "#27ae60" for _ in feature_names]

    ax.barh(
        [feature_names[j] for j in sorted_idx],
        mean_shap[sorted_idx],
        color="#2980b9"
    )
    ax.set_title(f"{label} Risk", fontsize=12)
    ax.set_xlabel("Mean |SHAP value|")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
bar_path = os.path.join(REPORT_DIR, "shap_bar_chart.png")
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"    Saved: {bar_path}")

# ------------------------------------------------------------------ #
#  Plot 2 — Beeswarm / dot plot per class                            #
# ------------------------------------------------------------------ #

for i, label in enumerate(class_labels):
    print(f"\n[4.{i+1}] Beeswarm plot — {label} Risk ...")
    plt.figure(figsize=(10, 5))
    shap.summary_plot(
        shap_values[i],        # ← FIX: index by class, not pass all
        X_sample,
        feature_names=feature_names,
        show=False,
        plot_type="dot"
    )
    plt.title(f"SHAP Values — {label} Risk Class",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    dot_path = os.path.join(REPORT_DIR,
                            f"shap_beeswarm_{label.lower()}_risk.png")
    plt.savefig(dot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"    Saved: {dot_path}")

# ------------------------------------------------------------------ #
#  Plot 3 — Single patient waterfall explanation                     #
# ------------------------------------------------------------------ #

print("\n[5] Waterfall plot — single high-risk patient explanation ...")

# Find a high-risk sample in the dataset
high_risk_idx = df[df["risk"] == 2].index
if len(high_risk_idx) > 0:
    sample_idx = X_sample.index.get_indexer([high_risk_idx[0]])[0]
    if sample_idx >= 0:
        for i, label in enumerate(class_labels):
            exp = shap.Explanation(
                values       = shap_values[i][sample_idx],
                base_values  = explainer.expected_value[i],
                data         = X_sample.iloc[sample_idx].values,
                feature_names= feature_names
            )
            plt.figure(figsize=(10, 4))
            shap.waterfall_plot(exp, show=False)
            plt.title(f"Single Patient — {label} Risk SHAP Explanation",
                      fontsize=12)
            plt.tight_layout()
            wf_path = os.path.join(REPORT_DIR,
                                   f"shap_waterfall_{label.lower()}.png")
            plt.savefig(wf_path, dpi=150, bbox_inches="tight")
            plt.show()
            print(f"    Saved: {wf_path}")

# ------------------------------------------------------------------ #
#  Print text summary                                                 #
# ------------------------------------------------------------------ #

print("\n" + "=" * 50)
print("  Feature Importance Summary (Mean |SHAP|)")
print("=" * 50)
for i, label in enumerate(class_labels):
    mean_shap = np.abs(shap_values[i]).mean(axis=0)
    print(f"\n  {label} Risk:")
    for feat, val in sorted(zip(feature_names, mean_shap),
                            key=lambda x: -x[1]):
        bar = "█" * int(val * 60)
        print(f"    {feat:18s}: {val:.4f}  {bar}")

print("\n  All plots saved to:", REPORT_DIR)
print("=" * 50)
