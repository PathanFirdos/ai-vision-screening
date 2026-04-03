"""
behaviour_model.py
------------------
Generates synthetic behaviour dataset and trains
a Random Forest behaviour classifier.

Classes:
    0 = Cooperative           (honest child)
    1 = Guessing              (random answers)
    2 = Not paying attention  (looking away)
    3 = Intentional wrong     (purposely wrong)

Features (8):
    accuracy, avg_response_time, response_variance,
    repeat_consistency, gaze_focus_score,
    large_letter_accuracy, small_letter_accuracy,
    control_letter_accuracy, answer_entropy

Run from project root:
    python ml/behaviour_model.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "..", "data",   "behaviour_dataset.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "..", "models", "behaviour_model.pkl")

os.makedirs(os.path.dirname(DATA_PATH),  exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

FEATURE_COLS = [
    "accuracy",
    "avg_response_time",
    "response_variance",
    "repeat_consistency",
    "gaze_focus_score",
    "large_letter_accuracy",
    "small_letter_accuracy",
    "control_letter_accuracy",
    "answer_entropy",
]

BEHAVIOUR_NAMES = {
    0: "Cooperative",
    1: "Guessing",
    2: "Not paying attention",
    3: "Intentional wrong",
}

np.random.seed(42)


# ------------------------------------------------------------------ #
#  Dataset generation                                                 #
# ------------------------------------------------------------------ #

def _clip(val, lo, hi):
    return max(lo, min(hi, val))


def _gen_cooperative(n):
    """Genuine child trying their best."""
    rows = []
    for _ in range(n):
        rows.append([
            _clip(np.random.normal(78, 10), 50, 100),    # accuracy
            _clip(np.random.normal(1.5,  0.4), 0.8, 3.0),# avg_response_time
            _clip(np.random.normal(0.12, 0.05), 0.0, 0.5),# response_variance
            _clip(np.random.normal(0.92, 0.06), 0.7, 1.0),# repeat_consistency
            _clip(np.random.normal(0.88, 0.08), 0.6, 1.0),# gaze_focus_score
            _clip(np.random.normal(0.90, 0.08), 0.6, 1.0),# large_letter_accuracy
            _clip(np.random.normal(0.55, 0.15), 0.2, 0.9),# small_letter_accuracy
            _clip(np.random.normal(0.92, 0.08), 0.6, 1.0),# control_letter_accuracy
            _clip(np.random.normal(0.25, 0.08), 0.1, 0.5),# answer_entropy
            0,  # label
        ])
    return rows


def _gen_guessing(n):
    """Child giving random answers quickly."""
    rows = []
    for _ in range(n):
        rows.append([
            _clip(np.random.normal(12,  5),   5,  30),    # accuracy ~11% random
            _clip(np.random.normal(0.30, 0.08), 0.1, 0.5),# very fast
            _clip(np.random.normal(0.02, 0.01), 0.0, 0.1),# very uniform timing
            _clip(np.random.normal(0.45, 0.10), 0.2, 0.7),# inconsistent repeats
            _clip(np.random.normal(0.55, 0.15), 0.2, 0.9),# gaze sometimes off
            _clip(np.random.normal(0.12, 0.06), 0.0, 0.3),# large=wrong too
            _clip(np.random.normal(0.12, 0.06), 0.0, 0.3),# small=wrong too
            _clip(np.random.normal(0.12, 0.06), 0.0, 0.3),# control=wrong
            _clip(np.random.normal(0.90, 0.05), 0.7, 1.0),# high entropy = uniform
            1,  # label
        ])
    return rows


def _gen_not_paying_attention(n):
    """Child looking away, not focusing."""
    rows = []
    for _ in range(n):
        rows.append([
            _clip(np.random.normal(40,  15), 15,  70),    # mediocre accuracy
            _clip(np.random.normal(2.8,  0.8), 0.8, 5.0), # slow (distracted)
            _clip(np.random.normal(0.50, 0.20), 0.1, 1.5),# high variance
            _clip(np.random.normal(0.60, 0.15), 0.3, 0.9),# inconsistent
            _clip(np.random.normal(0.30, 0.12), 0.1, 0.6),# low gaze ← key signal
            _clip(np.random.normal(0.55, 0.15), 0.2, 0.9),# large ok sometimes
            _clip(np.random.normal(0.25, 0.12), 0.0, 0.6),# small mostly wrong
            _clip(np.random.normal(0.50, 0.20), 0.1, 0.9),# control uncertain
            _clip(np.random.normal(0.60, 0.12), 0.3, 0.9),# moderate entropy
            2,  # label
        ])
    return rows


def _gen_intentional_wrong(n):
    """Child deliberately answering wrong."""
    rows = []
    for _ in range(n):
        rows.append([
            _clip(np.random.normal(15,   8),  3,  35),    # low overall accuracy
            _clip(np.random.normal(0.55, 0.15), 0.3, 1.2),# medium-fast
            _clip(np.random.normal(0.08, 0.04), 0.0, 0.3),# consistent (deliberate)
            _clip(np.random.normal(0.40, 0.12), 0.2, 0.7),# inconsistent on repeat
            _clip(np.random.normal(0.75, 0.12), 0.4, 1.0),# gaze OK (they look!)
            _clip(np.random.normal(0.05, 0.04), 0.0, 0.2),# large=WRONG ← key
            _clip(np.random.normal(0.75, 0.12), 0.4, 1.0),# small=correct ← key
            _clip(np.random.normal(0.08, 0.05), 0.0, 0.3),# control=wrong
            _clip(np.random.normal(0.45, 0.10), 0.2, 0.7),# moderate entropy
            3,  # label
        ])
    return rows


def generate_dataset(n_per_class=2500):
    print("[1] Generating behaviour dataset ...")
    rows = (
        _gen_cooperative(n_per_class) +
        _gen_guessing(n_per_class) +
        _gen_not_paying_attention(n_per_class) +
        _gen_intentional_wrong(n_per_class)
    )
    cols = FEATURE_COLS + ["behaviour"]
    df   = pd.DataFrame(rows, columns=cols)
    df   = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATA_PATH, index=False)

    print(f"    Saved: {DATA_PATH}")
    print(f"    Total: {len(df)} rows")
    print("    Distribution:")
    for cls, name in BEHAVIOUR_NAMES.items():
        n = (df["behaviour"]==cls).sum()
        print(f"      {name:25s}: {n}")
    return df


# ------------------------------------------------------------------ #
#  Training                                                           #
# ------------------------------------------------------------------ #

def train(df=None):
    if df is None:
        if not os.path.exists(DATA_PATH):
            df = generate_dataset()
        else:
            df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLS]
    y = df["behaviour"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[2] Training behaviour classifier ...")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Cross-validation
    cv = cross_val_score(model, X, y, cv=5, scoring="f1_weighted")
    print(f"    Cross-val F1: {cv.mean():.4f} ± {cv.std():.4f}")

    # Evaluation
    y_pred = model.predict(X_test)
    print("\n[3] Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=list(BEHAVIOUR_NAMES.values())
    ))

    # Feature importance
    print("[4] Feature Importances:")
    for feat, imp in sorted(zip(FEATURE_COLS, model.feature_importances_),
                            key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"    {feat:30s}: {imp:.4f}  {bar}")

    # Save
    bundle = {
        "model":          model,
        "feature_names":  FEATURE_COLS,
        "behaviour_names": BEHAVIOUR_NAMES,
        "classes":        list(model.classes_),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\n[5] Model saved: {MODEL_PATH}")

    return model


# ------------------------------------------------------------------ #
#  Quick predict utility                                              #
# ------------------------------------------------------------------ #

def predict_behaviour(features: dict) -> dict:
    """
    Predict behaviour class from feature dict.
    Called by ReliabilityScorer.

    Returns:
        dict with class, label, confidence, probabilities
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "behaviour_model.pkl not found. Run: python ml/behaviour_model.py"
        )

    bundle = joblib.load(MODEL_PATH)
    model  = bundle["model"]
    fnames = bundle["feature_names"]
    bnames = bundle["behaviour_names"]

    X   = pd.DataFrame([[features.get(f, 0.5) for f in fnames]],
                       columns=fnames)
    cls = int(model.predict(X)[0])
    prb = model.predict_proba(X)[0]

    return {
        "behaviour_class": cls,
        "behaviour_label": bnames[cls],
        "confidence":      round(float(prb[cls])*100, 1),
        "probabilities": {
            bnames[i]: round(float(p)*100, 1)
            for i, p in enumerate(prb)
        },
    }


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 50)
    print("  Behaviour Classifier — Training")
    print("=" * 50)

    df = generate_dataset(n_per_class=2500)
    train(df)

    # Quick test
    print("\n[6] Quick prediction test:")
    test_cases = [
        {"name": "Cooperative child",
         "accuracy":0.82,"avg_response_time":1.4,"response_variance":0.10,
         "repeat_consistency":0.95,"gaze_focus_score":0.90,
         "large_letter_accuracy":0.92,"small_letter_accuracy":0.55,
         "control_letter_accuracy":0.95,"answer_entropy":0.22},
        {"name": "Guessing child",
         "accuracy":0.12,"avg_response_time":0.28,"response_variance":0.02,
         "repeat_consistency":0.40,"gaze_focus_score":0.55,
         "large_letter_accuracy":0.10,"small_letter_accuracy":0.12,
         "control_letter_accuracy":0.10,"answer_entropy":0.92},
        {"name": "Intentional wrong",
         "accuracy":0.08,"avg_response_time":0.60,"response_variance":0.07,
         "repeat_consistency":0.38,"gaze_focus_score":0.78,
         "large_letter_accuracy":0.04,"small_letter_accuracy":0.82,
         "control_letter_accuracy":0.05,"answer_entropy":0.44},
    ]

    for tc in test_cases:
        name = tc.pop("name")
        res  = predict_behaviour(tc)
        print(f"  {name:28s}: {res['behaviour_label']} "
              f"({res['confidence']}% confidence)")

    print("\n" + "=" * 50)