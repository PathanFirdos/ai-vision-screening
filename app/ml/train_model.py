"""
train_model.py
--------------
Trains a RandomForestClassifier on the vision screening dataset.

Fixes applied:
  - oob_score=True added (free out-of-bag validation)
  - sklearn version + training date saved in model bundle
  - Sample prediction test run after saving
  - class_weight='balanced' for imbalanced classes
  - Feature names saved (fixes sklearn version warning)
  - Confusion matrix + full classification report printed

Run from project root:
    python ml/train_model.py
"""

import os
import sklearn
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.metrics           import classification_report, confusion_matrix

# ------------------------------------------------------------------ #
#  Paths                                                              #
# ------------------------------------------------------------------ #

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "data",   "vision_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "saved_model.pkl")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

FEATURE_COLS = [
    "accuracy",
    "response_time",
    "blink_rate",
    "gaze_stability",
    "distance",
]
TARGET_COL = "risk"
RISK_NAMES = {0: "Low", 1: "Moderate", 2: "High"}

# ------------------------------------------------------------------ #
#  1. Load dataset                                                    #
# ------------------------------------------------------------------ #

print("=" * 55)
print("  Vision Screening — Model Training")
print("=" * 55)
print(f"  sklearn version : {sklearn.__version__}")
print(f"  Training date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\n[ERROR] Dataset not found: {DATA_PATH}\n"
        "  Run this first:  python ml/generate_dataset.py\n"
    )

df = pd.read_csv(DATA_PATH)
print(f"[1] Dataset loaded")
print(f"    File    : {DATA_PATH}")
print(f"    Shape   : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"    NaN     : {df.isnull().sum().sum()} (should be 0)")
print()
print("    Class distribution:")
for label, name in RISK_NAMES.items():
    n   = (df[TARGET_COL] == label).sum()
    pct = n / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"      {name:10s}: {n:5d} ({pct:4.1f}%)  {bar}")

# ------------------------------------------------------------------ #
#  2. Features and target                                             #
# ------------------------------------------------------------------ #

X = df[FEATURE_COLS]
y = df[TARGET_COL]

print()
print("[2] Feature summary:")
for feat in FEATURE_COLS:
    print(f"    {feat:18s}: "
          f"mean={X[feat].mean():.3f}  std={X[feat].std():.3f}  "
          f"min={X[feat].min():.3f}  max={X[feat].max():.3f}")

# ------------------------------------------------------------------ #
#  3. Train / test split (stratified)                                 #
# ------------------------------------------------------------------ #

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # keep class ratios in both splits
)

print()
print(f"[3] Train/test split (80/20 stratified)")
print(f"    Train : {len(X_train)} rows")
print(f"    Test  : {len(X_test)} rows")
print(f"    Train class counts: {dict(y_train.value_counts().sort_index())}")
print(f"    Test  class counts: {dict(y_test.value_counts().sort_index())}")

# ------------------------------------------------------------------ #
#  4. Train model                                                     #
# ------------------------------------------------------------------ #

print()
print("[4] Training RandomForestClassifier ...")
print("    n_estimators : 200")
print("    class_weight : balanced  (handles class imbalance)")
print("    oob_score    : True      (free out-of-bag validation)")
print("    n_jobs       : -1        (all CPU cores)")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",   # handles unequal class sizes
    oob_score=True,            # FIX: free validation using unused samples
    n_jobs=-1,
)

model.fit(X_train, y_train)

print()
print(f"    Training complete.")
print(f"    OOB Score : {model.oob_score_:.4f}  "
      f"(out-of-bag accuracy — no test data used)")

# ------------------------------------------------------------------ #
#  5. Cross-validation                                                #
# ------------------------------------------------------------------ #

print()
print("[5] Cross-validation (5-fold, F1 weighted) ...")
cv_scores = cross_val_score(
    model, X, y, cv=5, scoring="f1_weighted", n_jobs=-1
)
print(f"    F1 scores  : {[round(s,4) for s in cv_scores]}")
print(f"    Mean F1    : {cv_scores.mean():.4f}")
print(f"    Std F1     : {cv_scores.std():.4f}")

# ------------------------------------------------------------------ #
#  6. Evaluate on held-out test set                                   #
# ------------------------------------------------------------------ #

y_pred = model.predict(X_test)

print()
print("[6] Test set evaluation:")
print()
print(classification_report(
    y_test, y_pred,
    target_names=["Low Risk", "Moderate Risk", "High Risk"],
    digits=4
))

print("    Confusion Matrix:")
cm     = confusion_matrix(y_test, y_pred)
header = f"{'':14s} {'Pred Low':>10s} {'Pred Mod':>10s} {'Pred High':>10s}"
print(f"    {header}")
for i, name in enumerate(["True Low", "True Mod", "True High"]):
    print(f"    {name:14s} {cm[i][0]:>10d} {cm[i][1]:>10d} {cm[i][2]:>10d}")

# ------------------------------------------------------------------ #
#  7. Feature importance                                              #
# ------------------------------------------------------------------ #

print()
print("[7] Feature importances (mean decrease in impurity):")
for feat, imp in sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"    {feat:18s}: {imp:.4f}  {bar}")

print()
print("    NOTE: accuracy and response_time are most important.")
print("    distance has lower importance (~6%) — it's a positioning")
print("    feature, not a vision quality indicator, but still useful")
print("    as patients with poor vision often sit too close.")

# ------------------------------------------------------------------ #
#  8. Save model bundle                                               #
# ------------------------------------------------------------------ #

save_bundle = {
    # Core model
    "model":           model,
    "feature_names":   FEATURE_COLS,
    "risk_names":      RISK_NAMES,
    "classes":         list(model.classes_),

    # FIX: save versions so loading on different sklearn never warns
    "sklearn_version": sklearn.__version__,
    "numpy_version":   np.__version__,
    "trained_on":      datetime.now().isoformat(),

    # Performance summary saved for reference
    "oob_score":       round(model.oob_score_, 4),
    "cv_f1_mean":      round(float(cv_scores.mean()), 4),
    "cv_f1_std":       round(float(cv_scores.std()),  4),
}

joblib.dump(save_bundle, MODEL_PATH)

print()
print(f"[8] Model saved to: {MODEL_PATH}")
print(f"    Bundle contains: model, feature_names, risk_names,")
print(f"                     sklearn_version, trained_on, oob_score")

# ------------------------------------------------------------------ #
#  9. Quick prediction test (verify save/load works)                 #
# ------------------------------------------------------------------ #

print()
print("[9] Verifying saved model (reload and predict) ...")

bundle_check = joblib.load(MODEL_PATH)
model_check  = bundle_check["model"]
feat_names   = bundle_check["feature_names"]
risk_names   = bundle_check["risk_names"]

test_cases = [
    {
        "name":           "Healthy patient",
        "accuracy":       92.0,
        "response_time":  1.2,
        "blink_rate":     15,
        "gaze_stability": 0.90,
        "distance":       58.0,
        "expected":       "Low",
    },
    {
        "name":           "Moderate risk patient",
        "accuracy":       65.0,
        "response_time":  2.6,
        "blink_rate":     26,
        "gaze_stability": 0.62,
        "distance":       50.0,
        "expected":       "Moderate",
    },
    {
        "name":           "High risk patient",
        "accuracy":       38.0,
        "response_time":  3.5,
        "blink_rate":     34,
        "gaze_stability": 0.38,
        "distance":       30.0,
        "expected":       "High",
    },
]

all_correct = True
for tc in test_cases:
    name     = tc.pop("name")
    expected = tc.pop("expected")
    row      = pd.DataFrame([[tc[f] for f in feat_names]], columns=feat_names)
    pred_cls = int(model_check.predict(row)[0])
    pred_lbl = risk_names[pred_cls]
    proba    = model_check.predict_proba(row)[0]
    conf     = round(float(proba[pred_cls]) * 100, 1)
    status   = "PASS" if pred_lbl == expected else "FAIL"
    if status == "FAIL":
        all_correct = False
    print(f"    [{status}] {name:25s}: "
          f"predicted={pred_lbl:10s} expected={expected:10s} "
          f"confidence={conf}%")

print()
if all_correct:
    print("    All 3 prediction tests passed.")
else:
    print("    [WARNING] Some predictions did not match expected.")
    print("    This may happen with small datasets. Regenerate and retrain.")

# ------------------------------------------------------------------ #
#  Summary                                                            #
# ------------------------------------------------------------------ #

print()
print("=" * 55)
print("  Training Complete")
print("=" * 55)
print(f"  OOB Accuracy  : {model.oob_score_:.4f}")
print(f"  CV F1 (mean)  : {cv_scores.mean():.4f}")
print(f"  Model saved   : {MODEL_PATH}")
print()
print("  Next steps:")
print("    python ml/vision_score.py      ← test scoring")
print("    python ml/shap_explain.py      ← explain model")
print("    python ml/behaviour_model.py   ← train behaviour classifier")
print("    python -m app.vision.face_mesh ← run full screening")
print("=" * 55)