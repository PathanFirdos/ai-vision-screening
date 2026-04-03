"""
reliability_score.py
--------------------
Combines all behaviour detector signals into one reliability score (0–100).
Also runs the behaviour classifier to label the child's behaviour type.

Usage:
    from app.vision.reliability_score import ReliabilityScorer

    scorer = ReliabilityScorer()
    result = scorer.score(behaviour_report)
    print(result["reliability"])     # 0–100
    print(result["verdict"])         # "Valid" / "Repeat test"
    print(result["behaviour_label"]) # "Cooperative" / "Guessing" / ...
    print(result["voice_message"])   # what to say to the child
"""

import os
import joblib

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "behaviour_model.pkl")

# Reliability thresholds
THRESHOLD_VALID    = 60
THRESHOLD_MODERATE = 80

# Behaviour class labels
BEHAVIOUR_LABELS = {
    0: "Cooperative",
    1: "Guessing",
    2: "Not paying attention",
    3: "Intentional wrong",
}

# Voice messages per verdict
VOICE_MESSAGES = {
    "valid_high": "Great job! Your answers look very reliable.",
    "valid_mod":  "Your answers look mostly reliable. We will continue.",
    "repeat":     ("I noticed some unusual answer patterns. "
                   "Let us try that test one more time. "
                   "Please look carefully at each letter and say what you see."),
    "guessing":   ("It looks like you might be guessing some answers. "
                   "Please take your time and say the letter you actually see. "
                   "There are no wrong answers — just tell me what you can see."),
    "not_looking":("Please look directly at the letter on the screen "
                   "before saying your answer. "
                   "Take a moment to focus your eyes on the letter."),
    "intentional":("Please try your best to answer correctly. "
                   "This test is to help check your eyes and keep you healthy. "
                   "Let us try again together."),
}


class ReliabilityScorer:

    def __init__(self):
        self._model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self._model = joblib.load(MODEL_PATH)
            except Exception as e:
                print(f"[ReliabilityScorer] Could not load model: {e}")

    # ---------------------------------------------------------------- #

    def score(self, behaviour_report: dict) -> dict:
        """
        Compute reliability score from behaviour_detector report.

        Args:
            behaviour_report: output of BehaviourDetector.get_report()

        Returns:
            dict with reliability, verdict, behaviour_label, voice_message
        """
        # Start at 100, apply deductions
        raw_score = max(0, 100 - behaviour_report["total_deduction"])

        # Clamp
        reliability = round(max(0, min(100, raw_score)), 1)

        # Reliability band
        if reliability >= THRESHOLD_VALID:
            if reliability >= THRESHOLD_MODERATE:
                verdict = "Valid"
                vm_key  = "valid_high"
            else:
                verdict = "Valid"
                vm_key  = "valid_mod"
        else:
            verdict = "Repeat test"
            vm_key  = "repeat"

        # Behaviour classification (ML or rule-based)
        features      = behaviour_report.get("features", {})
        behaviour_cls = self._classify_behaviour(features)
        behaviour_lbl = BEHAVIOUR_LABELS.get(behaviour_cls, "Cooperative")

        # Override voice message based on behaviour type
        if verdict == "Repeat test":
            if behaviour_cls == 1:
                vm_key = "guessing"
            elif behaviour_cls == 2:
                vm_key = "not_looking"
            elif behaviour_cls == 3:
                vm_key = "intentional"

        # Build breakdown
        breakdown = {
            "repeat_deduction":  behaviour_report["repeat_consistency"]["deduction"],
            "speed_deduction":   behaviour_report["response_time"]["deduction"],
            "pattern_deduction": behaviour_report["difficulty_pattern"]["deduction"],
            "gaze_deduction":    behaviour_report["gaze_fixation"]["deduction"],
            "control_deduction": behaviour_report["control_letters"]["deduction"],
            "entropy_deduction": behaviour_report["entropy"]["deduction"],
            "total_deduction":   behaviour_report["total_deduction"],
        }

        return {
            "reliability":      reliability,
            "verdict":          verdict,
            "is_valid":         verdict == "Valid",
            "behaviour_class":  behaviour_cls,
            "behaviour_label":  behaviour_lbl,
            "voice_message":    VOICE_MESSAGES[vm_key],
            "breakdown":        breakdown,
        }

    # ---------------------------------------------------------------- #

    def _classify_behaviour(self, features: dict) -> int:
        """
        Use ML model if available, else rule-based fallback.
        Returns class int: 0=Cooperative, 1=Guessing,
                           2=Not paying attention, 3=Intentional wrong
        """
        if self._model and "model" in self._model:
            try:
                import pandas as pd
                feat_names = self._model.get("feature_names",
                    ["accuracy","avg_response_time","response_variance",
                     "repeat_consistency","gaze_focus_score",
                     "large_letter_accuracy","small_letter_accuracy",
                     "control_letter_accuracy","answer_entropy"])
                row = [[features.get(f, 0.5) for f in feat_names]]
                X   = pd.DataFrame(row, columns=feat_names)
                return int(self._model["model"].predict(X)[0])
            except Exception as e:
                print(f"[ReliabilityScorer] ML predict error: {e}")

        # Rule-based fallback
        return self._rule_based(features)

    def _rule_based(self, f: dict) -> int:
        """Simple rule-based behaviour classifier."""
        rt   = f.get("avg_response_time", 1.5)
        gaze = f.get("gaze_focus_score", 0.8)
        cons = f.get("repeat_consistency", 1.0)
        ent  = f.get("answer_entropy", 0.3)
        lacc = f.get("large_letter_accuracy", 1.0)
        sacc = f.get("small_letter_accuracy", 0.0)

        # Intentional wrong: large=wrong, small=correct
        if lacc < 0.3 and sacc > 0.7:
            return 3

        # Guessing: very fast + high entropy
        if rt < 0.5 and ent > 0.7:
            return 1

        # Not paying attention: low gaze + inconsistent
        if gaze < 0.5 and cons < 0.6:
            return 2

        return 0   # Cooperative

    # ---------------------------------------------------------------- #

    def format_report(self, score_result: dict) -> str:
        """Human-readable report string."""
        r = score_result
        lines = [
            f"Test reliability      : {r['reliability']}/100",
            f"Verdict               : {r['verdict']}",
            f"Behaviour             : {r['behaviour_label']}",
            "",
            "Deductions:",
        ]
        for k, v in r["breakdown"].items():
            if k != "total_deduction" and v > 0:
                label = k.replace("_deduction","").replace("_"," ").title()
                lines.append(f"  {label:22s}: -{v}")
        lines.append(f"  {'Total':22s}: -{r['breakdown']['total_deduction']}")
        return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Standalone test                                                    #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    from behaviour_detector import BehaviourDetector, ALL_LETTERS
    import random

    scorer = ReliabilityScorer()

    for scenario, answers_fn in [
        ("Cooperative child",
         lambda l, s: l),
        ("Guessing child",
         lambda l, s: random.choice(ALL_LETTERS)),
        ("Intentional wrong child",
         lambda l, s: (random.choice([x for x in ALL_LETTERS if x!=l])
                       if s in ("large","huge") else l)),
    ]:
        print(f"\n{'='*45}")
        print(f"  Scenario: {scenario}")
        print(f"{'='*45}")

        bd = BehaviourDetector()
        letters = ["E","F","P","T","O","Z","L","C","D","E","F","P"]
        sizes   = ["large","large","medium","medium","small","small",
                   "tiny","huge","large","large","medium","small"]

        for i,(letter,size) in enumerate(zip(letters,sizes)):
            ans = answers_fn(letter, size)
            rt  = (random.uniform(0.2,0.4) if "Guessing" in scenario
                   else random.uniform(0.8,2.5))
            gaze= (random.random()>0.4 if "Not" in scenario
                   else random.random()>0.1)
            bd.record(letter=letter, size=size, answer=ans,
                      response_time=rt, gaze_ok=gaze,
                      is_control=(size=="huge"), round_num=i+1)

        report  = bd.get_report()
        result  = scorer.score(report)
        print(scorer.format_report(result))
        print(f"\nVoice message: \"{result['voice_message']}\"")