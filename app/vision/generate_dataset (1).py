"""
generate_dataset.py
-------------------
Generates a synthetic vision screening dataset with 10,000 samples.

Fixes applied:
  - Class imbalance fixed: high-risk samples boosted to ~15%
  - Realistic feature ranges matching real screening output
  - Saves to data/vision_dataset.csv

Run from project root:
    python ml/generate_dataset.py
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "vision_dataset.csv")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ------------------------------------------------------------------ #
#  Generation config                                                  #
# ------------------------------------------------------------------ #

SAMPLES = 10000

# Target class distribution (fixes severe imbalance from original)
# Original: 66.8% low / 28.7% moderate / 4.5% high  ← model ignores high
# Fixed   : 50%   low / 30%   moderate / 20%  high   ← balanced enough
CLASS_SPLITS = {
    0: 5000,   # low risk
    1: 3000,   # moderate risk
    2: 2000,   # high risk
}

def generate_low_risk(n):
    """Healthy vision profile."""
    return pd.DataFrame({
        "accuracy":       np.clip(np.random.normal(85, 8),  65, 100, ),
        "response_time":  np.clip(np.random.normal(1.4, 0.3), 0.8, 2.0),
        "blink_rate":     np.clip(np.random.normal(16, 4),   8,  22 ),
        "gaze_stability": np.clip(np.random.normal(0.85, 0.08), 0.65, 1.0),
        "distance":       np.clip(np.random.normal(57, 8),   42, 75 ),
        "risk":           0
    }, index=range(n)) if False else _gen(n, 0)

def generate_moderate_risk(n):
    return _gen(n, 1)

def generate_high_risk(n):
    return _gen(n, 2)

def _gen(n, risk_class):
    rows = []
    for _ in range(n):
        if risk_class == 0:
            accuracy       = np.clip(np.random.normal(85,  8),  65, 100)
            response_time  = np.clip(np.random.normal(1.4, 0.3), 0.8, 2.0)
            blink_rate     = np.clip(np.random.normal(16,  4),   8,  22)
            gaze_stability = np.clip(np.random.normal(0.85,0.08), 0.65, 1.0)
            distance       = np.clip(np.random.normal(57,  8),   42, 75)

        elif risk_class == 1:
            accuracy       = np.clip(np.random.normal(68, 8),   45, 79)
            response_time  = np.clip(np.random.normal(2.3, 0.4), 1.8, 3.2)
            blink_rate     = np.clip(np.random.normal(24, 5),   18, 35)
            gaze_stability = np.clip(np.random.normal(0.65,0.10), 0.45, 0.79)
            distance       = np.clip(np.random.normal(52, 10),   35, 75)

        else:  # high risk
            accuracy       = np.clip(np.random.normal(45, 10),  20, 62)
            response_time  = np.clip(np.random.normal(3.0, 0.5), 2.0, 4.5)
            blink_rate     = np.clip(np.random.normal(30, 6),   20, 45)
            gaze_stability = np.clip(np.random.normal(0.45,0.12), 0.20, 0.60)
            distance       = np.clip(np.random.normal(38, 8),   20, 52)

        rows.append([accuracy, response_time, blink_rate,
                     gaze_stability, distance, risk_class])
    return rows

# ------------------------------------------------------------------ #
#  Build dataset                                                      #
# ------------------------------------------------------------------ #

all_rows = []
for risk_class, count in CLASS_SPLITS.items():
    all_rows.extend(_gen(count, risk_class))

df = pd.DataFrame(all_rows, columns=[
    "accuracy", "response_time", "blink_rate",
    "gaze_stability", "distance", "risk"
])

# Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df.to_csv(OUTPUT_PATH, index=False)

# ------------------------------------------------------------------ #
#  Report                                                             #
# ------------------------------------------------------------------ #

print("=" * 45)
print("  Dataset Generated Successfully")
print("=" * 45)
print(f"  Saved to  : {OUTPUT_PATH}")
print(f"  Total rows: {len(df)}")
print()
print("  Class Distribution:")
for label, name in [(0,"Low Risk"), (1,"Moderate Risk"), (2,"High Risk")]:
    count = (df["risk"] == label).sum()
    pct   = count / len(df) * 100
    print(f"    {name:15s}: {count:5d}  ({pct:.1f}%)")
print()
print("  Feature Ranges:")
print(df.drop("risk", axis=1).describe().round(2).to_string())
print("=" * 45)
