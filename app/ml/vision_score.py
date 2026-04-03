"""
vision_score.py
---------------
Scores a patient's vision screening results using:
  1. Rule-based vision score (0–100)
  2. ML model risk prediction (Low / Moderate / High)

Fixes applied:
  - Score normalized to true 0–100 range (original min was 25)
  - Risk label returned alongside score
  - ML model prediction integrated
  - Input validation added
  - Connects directly to face_mesh.py screening output

Usage inside face_mesh.py:
    from ml.vision_score import score_patient
    result = score_patient(features_dict)
    print(result["score"], result["risk_label"])

Standalone test:
    python ml/vision_score.py
"""

import os
import numpy as np

# ------------------------------------------------------------------ #
#  Paths                                                              #
# ------------------------------------------------------------------ #

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "saved_model.pkl")

# ------------------------------------------------------------------ #
#  Load model (lazy — only when needed)                               #
# ------------------------------------------------------------------ #

_model_bundle = None

def _load_model():
    global _model_bundle
    if _model_bundle is None:
        try:
            import joblib
            _model_bundle = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"[WARNING] Could not load ML model: {e}")
            _model_bundle = {}
    return _model_bundle


# ------------------------------------------------------------------ #
#  Rule-based vision score                                            #
# ------------------------------------------------------------------ #

# Score weights (must sum to 100)
WEIGHTS = {
    "accuracy":        40,   # most important — letter identification
    "gaze_stability":  20,   # eye control
    "blink_rate":      15,   # eye health / fatigue
    "response_time":   15,   # cognitive speed
    "distance":        10,   # positioning
}

def _compute_raw_score(features: dict) -> float:
    """
    Compute raw score before normalization.
    Raw range: 25–100 (original had this bug — min was never 0).
    """
    score = 0.0

    # Accuracy: 0–100 → contributes 0–40
    score += (features["accuracy"] / 100.0) * WEIGHTS["accuracy"]

    # Gaze stability: 0–1 → contributes 0–20
    score += features["gaze_stability"] * WEIGHTS["gaze_stability"]

    # Blink rate: ideal 12–20, penalty for too high
    if features["blink_rate"] <= 20:
        score += WEIGHTS["blink_rate"]             # full 15
    elif features["blink_rate"] <= 28:
        score += WEIGHTS["blink_rate"] * 0.7       # 10.5
    else:
        score += WEIGHTS["blink_rate"] * 0.4       # 6

    # Response time: faster = better
    if features["response_time"] < 1.5:
        score += WEIGHTS["response_time"]          # full 15
    elif features["response_time"] < 2.5:
        score += WEIGHTS["response_time"] * 0.7    # 10.5
    else:
        score += WEIGHTS["response_time"] * 0.3    # 4.5

    # Distance: ideal 45–70 cm
    if 45 <= features["distance"] <= 70:
        score += WEIGHTS["distance"]               # full 10
    elif 35 <= features["distance"] <= 80:
        score += WEIGHTS["distance"] * 0.5         # 5
    else:
        score += 0                                 # out of range

    return score


RAW_MIN = 0 + 0 + 6 + 4.5 + 0     # = 10.5  (worst possible)
RAW_MAX = 40 + 20 + 15 + 15 + 10  # = 100.0 (best possible)


def compute_vision_score(features: dict) -> float:
    """
    Returns normalized vision score from 0 to 100.

    Args:
        features: dict with keys:
            accuracy (float 0–100)
            response_time (float seconds)
            blink_rate (float blinks/session)
            gaze_stability (float 0–1)
            distance (float cm)

    Returns:
        float: score from 0 to 100
    """
    raw   = _compute_raw_score(features)
    # Normalize to 0–100
    score = ((raw - RAW_MIN) / (RAW_MAX - RAW_MIN)) * 100
    return round(np.clip(score, 0, 100), 1)


# ------------------------------------------------------------------ #
#  ML model prediction                                                #
# ------------------------------------------------------------------ #

def predict_risk_ml(features: dict) -> dict:
    """
    Predict risk using the trained RandomForest model.

    Returns:
        dict with:
            risk_class (int): 0=Low, 1=Moderate, 2=High
            risk_label (str): "Low" / "Moderate" / "High"
            confidence (float): probability of predicted class
            probabilities (dict): all class probabilities
    """
    bundle = _load_model()

    if not bundle or "model" not in bundle:
        # Fallback to rule-based if model unavailable
        return _rule_based_risk(features)

    model         = bundle["model"]
    feature_names = bundle.get("feature_names",
                    ["accuracy","response_time","blink_rate",
                     "gaze_stability","distance"])
    risk_names    = bundle.get("risk_names", {0:"Low",1:"Moderate",2:"High"})

    import pandas as pd
    X = pd.DataFrame([[features[f] for f in feature_names]],
                     columns=feature_names)

    risk_class  = int(model.predict(X)[0])
    proba       = model.predict_proba(X)[0]
    confidence  = round(float(proba[risk_class]) * 100, 1)

    return {
        "risk_class":    risk_class,
        "risk_label":    risk_names[risk_class],
        "confidence":    confidence,
        "probabilities": {
            risk_names[i]: round(float(p) * 100, 1)
            for i, p in enumerate(proba)
        }
    }


def _rule_based_risk(features: dict) -> dict:
    """Fallback rule-based risk when ML model is not available."""
    acc  = features["accuracy"]
    gs   = features["gaze_stability"]
    dist = features["distance"]
    rt   = features["response_time"]
    br   = features["blink_rate"]

    if acc < 50 or gs < 0.5 or dist < 35:
        risk_class = 2
    elif acc < 70 or rt > 2.5 or br > 28:
        risk_class = 1
    else:
        risk_class = 0

    names = {0: "Low", 1: "Moderate", 2: "High"}
    return {
        "risk_class":    risk_class,
        "risk_label":    names[risk_class],
        "confidence":    None,
        "probabilities": None,
    }


# ------------------------------------------------------------------ #
#  Main function — combines score + ML prediction                    #
# ------------------------------------------------------------------ #

def score_patient(features: dict) -> dict:
    """
    Full patient scoring: rule-based score + ML risk prediction.

    Args:
        features: dict with keys:
            accuracy, response_time, blink_rate,
            gaze_stability, distance

    Returns:
        dict with:
            score (float):       0–100 vision score
            grade (str):         A / B / C / D / F
            risk_class (int):    0 / 1 / 2
            risk_label (str):    Low / Moderate / High
            confidence (float):  ML confidence %
            probabilities (dict)
            recommendations (list[str])
    """
    # Validate input
    required = ["accuracy","response_time","blink_rate",
                "gaze_stability","distance"]
    for key in required:
        if key not in features:
            raise ValueError(f"Missing feature: '{key}'")

    score  = compute_vision_score(features)
    ml_res = predict_risk_ml(features)
    grade  = _grade(score)
    recs   = _recommendations(score, ml_res["risk_class"], features)

    return {
        "score":         score,
        "grade":         grade,
        "risk_class":    ml_res["risk_class"],
        "risk_label":    ml_res["risk_label"],
        "confidence":    ml_res["confidence"],
        "probabilities": ml_res["probabilities"],
        "recommendations": recs,
    }


def _grade(score: float) -> str:
    if score >= 85: return "A"
    if score >= 70: return "B"
    if score >= 55: return "C"
    if score >= 40: return "D"
    return "F"


def _recommendations(score, risk_class, features) -> list:
    recs = []
    if risk_class == 2:
        recs.append("Consult an ophthalmologist for a full eye examination.")
    if features["accuracy"] < 60:
        recs.append("Low letter recognition accuracy — vision clarity test recommended.")
    if features["gaze_stability"] < 0.55:
        recs.append("Poor gaze stability detected — eye muscle evaluation advised.")
    if features["response_time"] > 2.8:
        recs.append("Slow response time — cognitive and visual processing check advised.")
    if features["blink_rate"] > 28:
        recs.append("High blink rate — possible eye strain or dry eye condition.")
    if features["distance"] < 35:
        recs.append("Screen too close during test — maintain 40–70 cm distance.")
    if not recs:
        recs.append("No significant concerns. Continue annual eye checkups.")
    return recs


# ------------------------------------------------------------------ #
#  Standalone test                                                    #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    test_cases = [
        {
            "name": "Healthy Patient",
            "accuracy": 92, "response_time": 1.2,
            "blink_rate": 15, "gaze_stability": 0.90, "distance": 58
        },
        {
            "name": "Moderate Risk",
            "accuracy": 65, "response_time": 2.6,
            "blink_rate": 26, "gaze_stability": 0.62, "distance": 50
        },
        {
            "name": "High Risk",
            "accuracy": 38, "response_time": 3.5,
            "blink_rate": 34, "gaze_stability": 0.38, "distance": 30
        },
    ]

    print("=" * 55)
    print("  Vision Score — Patient Test Results")
    print("=" * 55)

    for tc in test_cases:
        name = tc.pop("name")
        result = score_patient(tc)
        print(f"\n  Patient  : {name}")
        print(f"  Score    : {result['score']}/100  (Grade {result['grade']})")
        print(f"  Risk     : {result['risk_label']}", end="")
        if result["confidence"]:
            print(f"  (confidence: {result['confidence']}%)")
        else:
            print()
        if result["probabilities"]:
            for k, v in result["probabilities"].items():
                print(f"             {k}: {v}%")
        print("  Recommendations:")
        for r in result["recommendations"]:
            print(f"    • {r}")

    print("\n" + "=" * 55)
