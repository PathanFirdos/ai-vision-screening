"""
mic_live.py
-----------
Shows live what the microphone hears on the camera screen.
No typing needed. Just speak and watch.

FIXES APPLIED:
  - energy_threshold = 50  (your ambient noise is only 30)
  - device_index = 1       (Intel Smart Sound Array)
  - language = en-IN       (Indian English)
  - phrase_time_limit = 5  (longer capture window)

cd C:\projects\ai-vision-screening
python mic_live.py
"""

import cv2
import threading
import time
import queue
import speech_recognition as sr

# ── Letter matcher ───────────────────────────────────────────────────
VALID = {"C", "D", "E", "F", "L", "O", "P", "T", "Z"}

PHONETIC_MAP = {
    "c": "C", "see": "C", "sea": "C",
    "d": "D", "dee": "D",
    "e": "E", "ee": "E", "he": "E", "me": "E", "be": "E",
    "eat": "E", "heat": "E", "three": "E",
    "f": "F", "ef": "F", "eff": "F", "half": "F", "off": "F",
    "l": "L", "el": "L", "well": "L", "elle": "L",
    "o": "O", "oh": "O", "zero": "O", "no": "O", "low": "O",
    "p": "P", "pee": "P", "pay": "P",
    "t": "T", "tee": "T", "tea": "T", "two": "T", "to": "T",
    "z": "Z", "zee": "Z", "zed": "Z",
    "skip": "SKIP", "pass": "SKIP", "next": "SKIP",
    "letter c": "C", "letter d": "D", "letter e": "E",
    "letter f": "F", "letter l": "L", "letter o": "O",
    "letter p": "P", "letter t": "T", "letter z": "Z",
    "the letter c": "C", "the letter d": "D", "the letter e": "E",
    "the letter f": "F", "the letter l": "L", "the letter o": "O",
    "the letter p": "P", "the letter t": "T", "the letter z": "Z",
}


def match(text):
    if not text:
        return None
    t = text.strip().lower()
    if t.upper() in VALID:
        return t.upper()
    if t in PHONETIC_MAP:
        return PHONETIC_MAP[t]
    for word in t.split():
        if word.upper() in VALID:
            return word.upper()
        if word in PHONETIC_MAP:
            return PHONETIC_MAP[word]
    for phrase, letter in PHONETIC_MAP.items():
        if len(phrase) > 1 and phrase in t:
            return letter
    return None


# ── Mic listener thread ──────────────────────────────────────────────
class LiveMic:

    def __init__(self):
        self._q      = queue.Queue()
        self._status = "STARTING"
        self._lock   = threading.Lock()
        self._active = True
        threading.Thread(target=self._loop, daemon=True).start()

    def get_latest(self):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def status(self):
        with self._lock:
            return self._status

    def stop(self):
        self._active = False

    def _set(self, s):
        with self._lock:
            self._status = s

    def _recognize(self, rec, audio):
        """Try en-IN then en-US. Returns text or None."""
        for lang in ["en-IN", "en-US"]:
            try:
                return rec.recognize_google(audio, language=lang)
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"[API] {e}")
                return None
        return None

    def _loop(self):
        rec = sr.Recognizer()
        rec.energy_threshold         = 50
        rec.dynamic_energy_threshold = False
        rec.pause_threshold          = 0.8
        rec.phrase_threshold         = 0.1
        rec.non_speaking_duration    = 0.3

        self._set("READY")

        while self._active:
            try:
                with sr.Microphone(device_index=1) as source:
                    rec.adjust_for_ambient_noise(source, duration=0.3)
                    self._set("LISTENING")

                    try:
                        audio = rec.listen(
                            source,
                            timeout=None,
                            phrase_time_limit=5
                        )
                    except sr.WaitTimeoutError:
                        self._set("READY")
                        continue

                    self._set("PROCESSING")

                text    = self._recognize(rec, audio)
                matched = match(text) if text else None

                self._q.put({
                    "raw":     text or "...",
                    "matched": matched,
                    "time":    time.time(),
                })
                self._set("READY")

            except Exception as e:
                self._set("MIC ERROR")
                print(f"[Mic] {e}")
                time.sleep(1)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    mic = LiveMic()

    history     = []
    last_result = None
    last_time   = 0

    STATUS_COL = {
        "READY":      (0, 180, 0),
        "LISTENING":  (0, 220, 150),
        "PROCESSING": (200, 180, 0),
        "ERROR":      (0, 60, 255),
        "MIC ERROR":  (0, 0, 255),
        "STARTING":   (100, 100, 100),
    }

    print("=" * 55)
    print("  Live Mic Test")
    print("  Device  : Intel Smart Sound Array (index 1)")
    print("  Language: Indian English (en-IN)")
    print("  Press ESC to exit")
    print("=" * 55)
    print()
    print("  HOW TO SPEAK:")
    print("  Say 'EEE' and hold for 1 second — not just 'E'")
    print("  Or say 'the letter E' — works best")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        now   = time.time()

        result = mic.get_latest()
        if result:
            last_result = result
            last_time   = now
            history.append(result)
            if len(history) > 6:
                history.pop(0)
            m = result["matched"] or "NO MATCH"
            print(f"  Heard: '{result['raw']}'  ->  Matched: {m}")

        # Bottom panel
        cv2.rectangle(frame, (0, h - 295), (w, h), (10, 10, 10), -1)
        cv2.line(frame, (0, h - 295), (w, h - 295), (60, 60, 60), 1)

        # Title + status dot
        cv2.putText(frame, "LIVE MIC TEST",
                    (12, h - 268),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        status = mic.status()
        scol   = STATUS_COL.get(status, (150, 150, 150))
        if status != "LISTENING" or int(now * 2) % 2 == 0:
            cv2.circle(frame, (w - 30, h - 268), 8, scol, -1)
        cv2.putText(frame, f"Mic: {status}",
                    (12, h - 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, scol, 2)

        # Heard + matched
        if last_result and (now - last_time) < 5.0:
            cv2.putText(frame, "Heard:",
                        (12, h - 205),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 150, 150), 1)
            cv2.putText(frame, f'"{last_result["raw"]}"',
                        (88, h - 205),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 200), 2)

            m = last_result["matched"]
            if m and m in VALID:
                cv2.putText(frame, "Matched:",
                            (12, h - 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 150, 150), 1)
                cv2.putText(frame, m,
                            (115, h - 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 220, 80), 4)
            elif m == "SKIP":
                cv2.putText(frame, "-> SKIP",
                            (12, h - 158),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.80, (200, 180, 0), 2)
            else:
                cv2.putText(frame, "No match — say 'the letter E' slowly",
                            (12, h - 168),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 100, 255), 1)
        else:
            cv2.putText(frame, "Say: 'EEE'  or  'the letter E'",
                        (12, h - 188),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 120, 120), 1)
            cv2.putText(frame, "Hold sound for 1 full second",
                        (12, h - 162),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (90, 90, 90), 1)

        # History row
        cv2.putText(frame, "Recent:",
                    (12, h - 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        xp = 90
        for item in history[-6:]:
            m   = item["matched"] or "?"
            col = (0, 200, 80) if m in VALID else (0, 80, 200)
            cv2.putText(frame, m, (xp, h - 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            xp += 55

        # Top: valid letters
        cv2.putText(frame, "Valid letters:",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1)
        x2 = 132
        for lt in sorted(VALID):
            cv2.putText(frame, lt, (x2, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 255), 2)
            x2 += 44

        # Bottom instruction
        cv2.putText(frame,
                    "Say 'EEE' (hold 1 sec)  or  'the letter E'  |  ESC to exit",
                    (12, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1)

        cv2.imshow("Live Mic Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    mic.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()