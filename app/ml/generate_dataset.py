"""
generate_dataset.py
-------------------
Generates a synthetic vision screening dataset with 10,000 samples.

Fixes applied:
  - Removed 3 dead functions (generate_low/moderate/high_risk) never called
  - Removed unused SAMPLES variable
  - Added data validation (NaN check + range check)
  - Added noise layer so data is more realistic
  - sklearn version saved for reproducibility

Run from project root:
    python ml/generate_dataset.py
"""

import numpy as np
import pandas as pd
import os
import sklearn

np.random.seed(42)

# ------------------------------------------------------------------ #
#  Paths                                                              #
# ------------------------------------------------------------------ #

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "vision_dataset.csv")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ------------------------------------------------------------------ #
#  Class distribution (balanced — fixes original 4.5% high-risk)     #
# ------------------------------------------------------------------ #

CLASS_SPLITS = {
    0: 5000,   # low risk    50%
    1: 3000,   # moderate    30%
    2: 2000,   # high risk   20%
}

FEATURE_COLS = [
    "accuracy",        # % of letters identified correctly   (0–100)
    "response_time",   # average seconds per answer          (0.8–4.5)
    "blink_rate",      # total blinks during session         (8–45)
    "gaze_stability",  # fraction of time gaze on screen     (0–1)
    "distance",        # sitting distance from screen in cm  (20–80)
]

RISK_NAMES = {0: "Low", 1: "Moderate", 2: "High"}

# ------------------------------------------------------------------ #
#  Feature ranges per class                                           #
# ------------------------------------------------------------------ #

# Each class defined as (mean, std, min_clip, max_clip)
CLASS_PROFILES = {
    0: {   # Low risk — healthy vision
        "accuracy":       (85,  8,   65,  100),
        "response_time":  (1.4, 0.3, 0.8, 2.0),
        "blink_rate":     (16,  4,   8,   22 ),
        "gaze_stability": (0.85,0.08,0.65,1.0),
        "distance":       (57,  8,   42,  75 ),
    },
    1: {   # Moderate risk — mild vision issues
        "accuracy":       (68,  8,   45,  79 ),
        "response_time":  (2.3, 0.4, 1.8, 3.2),
        "blink_rate":     (24,  5,   18,  35 ),
        "gaze_stability": (0.65,0.10,0.45,0.79),
        "distance":       (52,  10,  35,  75 ),
    },
    2: {   # High risk — significant vision impairment
        "accuracy":       (45,  10,  20,  62 ),
        "response_time":  (3.0, 0.5, 2.0, 4.5),
        "blink_rate":     (30,  6,   20,  45 ),
        "gaze_stability": (0.45,0.12,0.20,0.60),
        "distance":       (38,  8,   20,  52 ),
    },
}


# ------------------------------------------------------------------ #
#  Single generator function (replaces 3 dead functions)             #
# ------------------------------------------------------------------ #

def _generate_class(n: int, risk_class: int) -> list:
    """
    Generate n rows for a given risk class.
    Adds small random noise on top of normal distribution
    to simulate real-world measurement variation.
    """
    profile = CLASS_PROFILES[risk_class]
    rows    = []

    for _ in range(n):
        row = []
        for feat in FEATURE_COLS:
            mean, std, lo, hi = profile[feat]
            # Base normal sample
            val = np.random.normal(mean, std)
            # Add small noise (+/- 2% extra variation for realism)
            val += np.random.normal(0, std * 0.05)
            # Clip to valid range
            val = float(np.clip(val, lo, hi))
            row.append(val)
        row.append(risk_class)
        rows.append(row)

    return rows


# ------------------------------------------------------------------ #
#  Build dataset                                                      #
# ------------------------------------------------------------------ #

def generate():
    all_rows = []
    for risk_class, count in CLASS_SPLITS.items():
        all_rows.extend(_generate_class(count, risk_class))

    df = pd.DataFrame(all_rows, columns=FEATURE_COLS + ["risk"])

    # Shuffle rows so classes are mixed
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


# ------------------------------------------------------------------ #
#  Validation                                                         #
# ------------------------------------------------------------------ #

def validate(df: pd.DataFrame) -> bool:
    """
    Validate generated dataset.
    Checks: no NaN, correct row count, all features in expected range.
    Returns True if all checks pass.
    """
    ok = True

    # NaN check
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        print(f"  [FAIL] NaN values found: {nulls}")
        ok = False
    else:
        print("  [PASS] No NaN values")

    # Row count check
    total_expected = sum(CLASS_SPLITS.values())
    if len(df) != total_expected:
        print(f"  [FAIL] Row count: expected {total_expected}, got {len(df)}")
        ok = False
    else:
        print(f"  [PASS] Row count: {len(df)}")

    # Feature range checks
    ranges = {
        "accuracy":       (0,   100),
        "response_time":  (0.5, 5.0),
        "blink_rate":     (5,   50 ),
        "gaze_stability": (0,   1.0),
        "distance":       (15,  85 ),
    }
    for feat, (lo, hi) in ranges.items():
        out = df[(df[feat] < lo) | (df[feat] > hi)]
        if len(out) > 0:
            print(f"  [FAIL] {feat}: {len(out)} values outside [{lo}, {hi}]")
            ok = False
        else:
            print(f"  [PASS] {feat}: all values in range [{lo}, {hi}]")

    # Class distribution check
    for cls, expected_n in CLASS_SPLITS.items():
        actual_n = (df["risk"] == cls).sum()
        if actual_n != expected_n:
            print(f"  [FAIL] Class {cls}: expected {expected_n}, got {actual_n}")
            ok = False
        else:
            print(f"  [PASS] Class {cls} ({RISK_NAMES[cls]}): {actual_n} rows")

    return ok


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 50)
    print("  Vision Dataset Generator")
    print("=" * 50)
    print(f"  sklearn version : {sklearn.__version__}")
    print(f"  numpy version   : {np.__version__}")
    print()

    print("[1] Generating dataset ...")
    df = generate()

    print("[2] Validating dataset ...")
    valid = validate(df)

    if not valid:
        print("\n[ERROR] Dataset validation failed. Check issues above.")
        exit(1)

    print()
    print("[3] Saving dataset ...")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"    Saved to: {OUTPUT_PATH}")

    print()
    print("[4] Dataset Summary:")
    print(f"    Total rows  : {len(df)}")
    print(f"    Total columns: {len(df.columns)}")
    print()
    print("    Class Distribution:")
    for label, name in RISK_NAMES.items():
        count = (df["risk"] == label).sum()
        pct   = count / len(df) * 100
        bar   = "█" * int(pct / 2)
        print(f"      {name:10s}: {count:5d} ({pct:4.1f}%)  {bar}")

    print()
    print("    Feature Statistics:")
    print(df[FEATURE_COLS].describe().round(3).to_string())

    print()
    print("    Feature Ranges per Class:")
    for cls, name in RISK_NAMES.items():
        sub = df[df["risk"] == cls]
        print(f"\n      {name} Risk:")
        for feat in FEATURE_COLS:
            print(f"        {feat:18s}: "
                  f"mean={sub[feat].mean():.2f}  "
                  f"std={sub[feat].std():.2f}  "
                  f"min={sub[feat].min():.2f}  "
                  f"max={sub[feat].max():.2f}")

    print()
    print("=" * 50)
    print("  Dataset generation complete.")
    print("  Next step: python ml/train_model.py")
    print("=" * 50)