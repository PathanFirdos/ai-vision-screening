"""
behaviour_detector.py
---------------------
Detects cheating / non-cooperation patterns during the letter test.

Tracks per-session:
  1. Repeat consistency    — same letter shown again, did answer change?
  2. Response time         — too fast = guessing, too slow = struggling
  3. Difficulty pattern    — large=wrong + small=correct = impossible
  4. Gaze fixation         — did child look at letter before speaking?
  5. Control letter miss   — huge obvious letter answered wrong
  6. Answer entropy        — too uniform distribution = guessing

Usage:
    from app.vision.behaviour_detector import BehaviourDetector

    bd = BehaviourDetector()
    bd.record(letter="E", size="large", answer="P",
              response_time=0.3, gaze_ok=True, is_control=False)
    report = bd.get_report()
"""

import math
import time
from collections import defaultdict


# Letter size categories
SIZE_ORDER = {"huge": 0, "large": 1, "medium": 2, "small": 3, "tiny": 4}

# Response time thresholds (seconds)
RT_GUESSING   = 0.4    # faster than this = likely guessing
RT_NORMAL_MIN = 0.8
RT_NORMAL_MAX = 3.0
RT_STRUGGLING = 5.0    # slower than this = struggling

# Gaze fixation minimum (seconds child must look at letter before answering)
GAZE_FIXATION_MIN = 0.30   # 300 ms

# All valid letters in the test
ALL_LETTERS = ["C","D","E","F","L","O","P","T","Z"]


class BehaviourDetector:
    """
    Records every answer during the letter test and detects
    suspicious behaviour patterns.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Call at start of each new test session."""
        self._records      = []         # list of answer dicts
        self._shown_map    = {}         # letter → list of (round, answer)
        self._answer_counts = defaultdict(int)  # for entropy
        self._session_start = time.time()

    # ---------------------------------------------------------------- #
    #  Main recording method                                             #
    # ---------------------------------------------------------------- #

    def record(self,
               letter: str,
               size: str,
               answer: str,
               response_time: float,
               gaze_ok: bool,
               is_control: bool = False,
               round_num: int = None):
        """
        Record one answer.

        Args:
            letter        : correct letter shown  (e.g. "E")
            size          : "huge"/"large"/"medium"/"small"/"tiny"
            answer        : what child said       (e.g. "P" or "SKIP")
            response_time : seconds from letter display to spoken answer
            gaze_ok       : True if child looked at letter ≥ 300ms
            is_control    : True if this is a hidden control letter
            round_num     : sequential round number (auto if None)
        """
        rn = round_num if round_num is not None else len(self._records) + 1
        correct = (answer.upper() == letter.upper())

        rec = dict(
            round         = rn,
            letter        = letter.upper(),
            size          = size,
            size_rank     = SIZE_ORDER.get(size, 2),
            answer        = answer.upper(),
            correct       = correct,
            response_time = response_time,
            gaze_ok       = gaze_ok,
            is_control    = is_control,
        )
        self._records.append(rec)

        # Track per-letter history for repeat consistency
        if letter not in self._shown_map:
            self._shown_map[letter] = []
        self._shown_map[letter].append((rn, answer.upper()))

        # Track answer distribution for entropy
        self._answer_counts[answer.upper()] += 1

    # ---------------------------------------------------------------- #
    #  Pattern detections                                               #
    # ---------------------------------------------------------------- #

    def _repeat_consistency_score(self) -> dict:
        """
        Check if same letter answered differently on repeat showing.
        Returns score component and list of inconsistent letters.
        """
        inconsistent = []
        for letter, history in self._shown_map.items():
            if len(history) < 2:
                continue
            answers = [h[1] for h in history]
            # If any two answers differ, it's inconsistent
            if len(set(answers)) > 1:
                # But if first=wrong and second=correct → they can see it
                # This is actually the CATCH: first was fake
                inconsistent.append({
                    "letter":  letter,
                    "answers": answers,
                    "caught":  (answers[0] != letter and
                                answers[-1] == letter)
                })
        return {
            "inconsistent_count": len(inconsistent),
            "details":            inconsistent,
            "deduction":          len(inconsistent) * 10,
        }

    def _response_time_score(self) -> dict:
        """Classify response times across all answers."""
        if not self._records:
            return {"guessing_count":0,"struggling_count":0,"deduction":0}

        guessing   = [r for r in self._records if r["response_time"] < RT_GUESSING]
        struggling = [r for r in self._records if r["response_time"] > RT_STRUGGLING]
        times      = [r["response_time"] for r in self._records]

        return {
            "avg_response_time":  round(sum(times)/len(times), 3),
            "response_variance":  round(_variance(times), 4),
            "guessing_count":     len(guessing),
            "struggling_count":   len(struggling),
            "deduction":          len(guessing) * 5,
        }

    def _difficulty_pattern_score(self) -> dict:
        """
        Detect impossible pattern: large letter wrong + small letter correct.
        This is biologically impossible in genuine vision loss.
        """
        large_correct = [r for r in self._records
                         if r["size_rank"] <= 1 and r["correct"]]
        large_wrong   = [r for r in self._records
                         if r["size_rank"] <= 1 and not r["correct"]]
        small_correct = [r for r in self._records
                         if r["size_rank"] >= 3 and r["correct"]]
        small_wrong   = [r for r in self._records
                         if r["size_rank"] >= 3 and not r["correct"]]

        impossible = (len(large_wrong) > 0 and
                      len(small_correct) > len(large_correct))

        total = len(self._records)
        large_acc = (len(large_correct) /
                     max(1, len(large_correct)+len(large_wrong))) * 100
        small_acc = (len(small_correct) /
                     max(1, len(small_correct)+len(small_wrong))) * 100

        return {
            "large_letter_accuracy": round(large_acc, 1),
            "small_letter_accuracy": round(small_acc, 1),
            "impossible_pattern":    impossible,
            "deduction":             15 if impossible else 0,
        }

    def _gaze_fixation_score(self) -> dict:
        """Check how many answers were given without looking at the letter."""
        gaze_miss = [r for r in self._records if not r["gaze_ok"]]
        total     = max(1, len(self._records))
        gaze_pct  = (total - len(gaze_miss)) / total * 100

        return {
            "gaze_focus_score":     round(gaze_pct, 1),
            "gaze_miss_count":      len(gaze_miss),
            "deduction":            len(gaze_miss) * 10,
        }

    def _control_letter_score(self) -> dict:
        """
        Control letters are huge obvious letters inserted secretly.
        Missing them = definitely not cooperating.
        """
        controls       = [r for r in self._records if r["is_control"]]
        control_wrong  = [r for r in controls if not r["correct"]]
        acc = (1 - len(control_wrong)/max(1, len(controls))) * 100

        return {
            "control_letter_accuracy": round(acc, 1),
            "control_wrong_count":     len(control_wrong),
            "deduction":               len(control_wrong) * 20,
        }

    def _entropy_score(self) -> dict:
        """
        If answer distribution is too uniform → likely guessing.
        Real patients cluster answers (mostly correct + a few near-misses).
        """
        total  = sum(self._answer_counts.values())
        if total == 0:
            return {"entropy":0,"uniform":False,"deduction":0}

        probs   = [c/total for c in self._answer_counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        max_ent = math.log2(len(ALL_LETTERS))  # = 3.17 for 9 letters
        norm_ent = entropy / max_ent            # 0=focused, 1=fully random

        # Random guessing → high entropy (>0.85)
        too_uniform = norm_ent > 0.85

        return {
            "entropy":        round(norm_ent, 3),
            "uniform":        too_uniform,
            "deduction":      10 if too_uniform else 0,
        }

    # ---------------------------------------------------------------- #
    #  8 features for behaviour classifier                              #
    # ---------------------------------------------------------------- #

    def get_features(self) -> dict:
        """
        Returns the 8 features used by the behaviour Random Forest.
        Call this at end of test session.
        """
        rt  = self._response_time_score()
        dp  = self._difficulty_pattern_score()
        gf  = self._gaze_fixation_score()
        rc  = self._repeat_consistency_score()
        cl  = self._control_letter_score()
        ent = self._entropy_score()

        total  = max(1, len(self._records))
        correct= sum(1 for r in self._records if r["correct"])

        return {
            "accuracy":               round(correct/total*100, 1),
            "avg_response_time":      rt["avg_response_time"],
            "response_variance":      rt["response_variance"],
            "repeat_consistency":     round(
                                        1 - rc["inconsistent_count"] /
                                        max(1, len(self._shown_map)), 3),
            "gaze_focus_score":       round(gf["gaze_focus_score"]/100, 3),
            "large_letter_accuracy":  round(dp["large_letter_accuracy"]/100, 3),
            "small_letter_accuracy":  round(dp["small_letter_accuracy"]/100, 3),
            "control_letter_accuracy":round(cl["control_letter_accuracy"]/100, 3),
            "answer_entropy":         ent["entropy"],
        }

    # ---------------------------------------------------------------- #
    #  Full session report                                              #
    # ---------------------------------------------------------------- #

    def get_report(self) -> dict:
        """
        Returns complete behaviour analysis report.
        """
        rc  = self._repeat_consistency_score()
        rt  = self._response_time_score()
        dp  = self._difficulty_pattern_score()
        gf  = self._gaze_fixation_score()
        cl  = self._control_letter_score()
        ent = self._entropy_score()

        total_deduction = (rc["deduction"] + rt["deduction"] +
                           dp["deduction"] + gf["deduction"] +
                           cl["deduction"] + ent["deduction"])

        return {
            "repeat_consistency":   rc,
            "response_time":        rt,
            "difficulty_pattern":   dp,
            "gaze_fixation":        gf,
            "control_letters":      cl,
            "entropy":              ent,
            "total_deduction":      total_deduction,
            "features":             self.get_features(),
            "records":              self._records,
        }


# ------------------------------------------------------------------ #
#  Utility                                                            #
# ------------------------------------------------------------------ #

def _variance(values: list) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean)**2 for v in values) / len(values)


# ------------------------------------------------------------------ #
#  Standalone test                                                    #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import random

    bd = BehaviourDetector()

    # Simulate a suspicious session
    letters = ["E","F","P","T","O","Z","L","C","D","E","F"]
    sizes   = ["large","large","medium","medium","small",
               "small","tiny","huge","large","large","medium"]

    print("Simulating suspicious child (intentional wrong answers)...")
    for i,(letter,size) in enumerate(zip(letters,sizes)):
        # Intentionally wrong on large, correct on small
        if size in ("large","huge"):
            answer = random.choice([l for l in ALL_LETTERS if l!=letter])
        else:
            answer = letter

        rt   = random.uniform(0.2, 0.5)   # very fast
        gaze = random.random() > 0.6       # often not looking

        bd.record(letter=letter, size=size, answer=answer,
                  response_time=rt, gaze_ok=gaze,
                  is_control=(size=="huge"), round_num=i+1)

    report = bd.get_report()
    print("\n=== Behaviour Report ===")
    print(f"Inconsistent repeats   : {report['repeat_consistency']['inconsistent_count']}")
    print(f"Guessing (fast) count  : {report['response_time']['guessing_count']}")
    print(f"Impossible pattern     : {report['difficulty_pattern']['impossible_pattern']}")
    print(f"Gaze focus score       : {report['gaze_fixation']['gaze_focus_score']}%")
    print(f"Control wrong          : {report['control_letters']['control_wrong_count']}")
    print(f"Answer entropy         : {report['entropy']['entropy']}")
    print(f"Total deduction        : {report['total_deduction']}")

    print("\n=== Behaviour Features ===")
    for k,v in report["features"].items():
        print(f"  {k:30s}: {v}")