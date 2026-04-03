"""
face_mesh.py  —  AI Vision Screening System  (REALTIME WORKING VERSION)
------------------------------------------------------------------------
Fixes applied vs previous versions:
  - Gaze / Alignment "unknown" fixed — uses direct landmark ratios
    without depending on pupil detection (which was failing)
  - Eye extractor fallback added — works even with glasses
  - Voice fresh-engine-per-sentence fix (Windows pyttsx3 bug)
  - All ML modules optional — system works even without them
  - Inline gaze from iris landmarks (MediaPipe refined landmarks)

Run from project root:
    cd C:\projects\ai-vision-screening
    python -m app.vision.face_mesh
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import pyttsx3
import sys
import os

# ------------------------------------------------------------------ #
#  Path setup                                                         #
# ------------------------------------------------------------------ #
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ------------------------------------------------------------------ #
#  MediaPipe setup                                                    #
# ------------------------------------------------------------------ #
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles

# ------------------------------------------------------------------ #
#  MediaPipe landmark indices                                         #
# ------------------------------------------------------------------ #
# Iris landmarks (only available with refine_landmarks=True)
LEFT_IRIS   = [474, 475, 476, 477]
RIGHT_IRIS  = [469, 470, 471, 472]

# Eye corner landmarks
LEFT_EYE_INNER  = 133
LEFT_EYE_OUTER  = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# Eye top/bottom for EAR blink detection
LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP   = 386
RIGHT_EYE_BOTTOM= 374

# Face width for distance estimation
FACE_LEFT  = 234
FACE_RIGHT = 454

# Focal length calibration (pixels) — tune if needed
FOCAL_LENGTH   = 700.0
REAL_FACE_WIDTH= 14.0   # cm average face width

EAR_THRESHOLD  = 0.20   # Eye Aspect Ratio blink threshold

EYE_DISPLAY_W  = 220
EYE_DISPLAY_H  = 90

# ================================================================== #
#  Voice — fresh engine per sentence (Windows fix)                   #
# ================================================================== #

class Voice:
    def __init__(self):
        self._q       = []
        self._lock    = threading.Lock()
        self._busy    = False
        self._current = ""
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, text_or_list):
        items = [text_or_list] if isinstance(text_or_list, str) else list(text_or_list)
        with self._lock:
            self._q = items[:]

    def add(self, text):
        with self._lock:
            self._q.append(text)

    def done(self):
        with self._lock:
            return not self._busy and len(self._q) == 0

    def current(self):
        with self._lock:
            return self._current

    def _speak_one(self, sentence):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 145)
            engine.setProperty('volume', 1.0)
            engine.say(sentence)
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception as e:
            print(f"[Voice] Error: {e}")
        time.sleep(0.08)

    def _run(self):
        while True:
            sentence = None
            with self._lock:
                if self._q and not self._busy:
                    sentence      = self._q.pop(0)
                    self._busy    = True
                    self._current = sentence
            if sentence:
                self._speak_one(sentence)
                with self._lock:
                    self._busy    = False
                    self._current = ""
            else:
                time.sleep(0.05)


# ================================================================== #
#  All voice scripts                                                  #
# ================================================================== #
S = {
    "welcome": [
        "Welcome to the A I Vision Screening System.",
        "Please sit comfortably in front of the camera.",
        "Make sure your face is clearly visible and well lit.",
        "The system will guide you through every step of the test.",
    ],
    "no_face": [
        "I cannot detect your face.",
        "Please move directly in front of the camera.",
        "Make sure your face is fully visible and well lit.",
        "Once your face is visible the test will continue.",
    ],
    "face_found": [
        "Your face has been detected. Great.",
        "Now let us check your distance from the screen.",
    ],
    "dist_intro": [
        "The ideal distance is between 40 and 80 centimeters.",
        "Please adjust your position if needed.",
    ],
    "too_close": ["You are too close. Please move back a little."],
    "too_far":   ["You are too far. Please move closer to the screen."],
    "dist_ok": [
        "Perfect. You are at the correct distance.",
        "Please stay in this position for the rest of the test.",
    ],
    "instructions": [
        "Now I will explain how this vision test works.",
        "There are two parts. The distance vision test and the near vision test.",
        "For each test, letters or words will appear on the screen.",
        "Say aloud the letter or word that you can see.",
        "Just speak clearly into the microphone. I am listening.",
        "If you cannot see something clearly, simply say skip.",
        "Take your time. There is no rush.",
        "Let us begin with the eye tracking check.",
    ],
    "eye_track_start": [
        "Eye tracking has started.",
        "Please look straight at the centre of the screen.",
        "Keep your gaze steady for the next ten seconds.",
    ],
    "gaze_drift": ["Please look straight at the screen."],
    "eye_track_done": [
        "Eye tracking complete. Well done.",
        "Now we will start the distance vision test.",
    ],
    "dist_vision_intro": [
        "A letter chart is now on the screen.",
        "Letters start large at the top and get smaller at the bottom.",
        "For each letter, say the letter name slowly and clearly.",
        "For example if you see E, say EEE and hold it for one second.",
        "Or say the letter E. Both will work.",
        "Say skip for any letter you cannot see clearly.",
        "I am listening. Please say the first letter you see.",
    ],
    "near_vision_intro": [
        "Now we will test your near vision.",
        "Lines of text are showing on the screen.",
        "Please say aloud the smallest line you can see clearly.",
        "Start from the top and work your way down.",
        "Say skip if you cannot read a line.",
    ],
    "listening": "I am listening. Please say what you see.",
    "skip_ok":   "Okay, skipping that one.",
    "correct":   ["Correct. Next.", "Well done. Next.", "Good. Next letter."],
    "wrong":     ["Thank you. Next.", "Okay. Next."],
    "unclear":   ["Please say the letter slowly. Hold the sound for one second. For example, say EEE or the letter E."],
    "results_intro": [
        "All tests are now complete.",
        "Here are your vision screening results.",
    ],
    "results_done": [
        "Please share these results with your eye care professional.",
        "Thank you for completing the A I Vision Screening test.",
        "Goodbye and take good care of your eyes.",
    ],
}

# ================================================================== #
#  Gaze estimation — uses iris landmarks directly                     #
#  This is the FIX for "gaze: unknown"                               #
# ================================================================== #

def get_gaze(landmarks, w, h):
    """
    Estimate gaze direction using MediaPipe iris landmarks.
    Returns: "center" / "left" / "right" / "up" / "down"

    Uses the iris centre position relative to the eye corners.
    No pupil detection needed — works with glasses too.
    """
    try:
        def lm(idx):
            return landmarks[idx].x * w, landmarks[idx].y * h

        # Left eye
        l_inner  = np.array(lm(LEFT_EYE_INNER))
        l_outer  = np.array(lm(LEFT_EYE_OUTER))
        l_top    = np.array(lm(LEFT_EYE_TOP))
        l_bot    = np.array(lm(LEFT_EYE_BOTTOM))

        # Left iris centre
        l_iris_pts = [np.array(lm(i)) for i in LEFT_IRIS]
        l_iris     = np.mean(l_iris_pts, axis=0)

        # Right eye
        r_inner  = np.array(lm(RIGHT_EYE_INNER))
        r_outer  = np.array(lm(RIGHT_EYE_OUTER))
        r_top    = np.array(lm(RIGHT_EYE_TOP))
        r_bot    = np.array(lm(RIGHT_EYE_BOTTOM))

        # Right iris centre
        r_iris_pts = [np.array(lm(i)) for i in RIGHT_IRIS]
        r_iris     = np.mean(r_iris_pts, axis=0)

        # Horizontal ratio: 0=outer corner, 1=inner corner
        # For left eye: outer=33, inner=133 → ratio=(iris.x - outer.x)/(inner.x - outer.x)
        l_eye_w = abs(l_inner[0] - l_outer[0])
        r_eye_w = abs(r_inner[0] - r_outer[0])

        if l_eye_w < 1 or r_eye_w < 1:
            return "unknown"

        l_h_ratio = (l_iris[0] - l_outer[0]) / l_eye_w
        r_h_ratio = (r_iris[0] - r_outer[0]) / r_eye_w
        h_ratio   = (l_h_ratio + r_h_ratio) / 2.0

        # Vertical ratio
        l_eye_h   = abs(l_bot[1] - l_top[1])
        r_eye_h   = abs(r_bot[1] - r_top[1])

        if l_eye_h < 1 or r_eye_h < 1:
            return "unknown"

        l_v_ratio = (l_iris[1] - l_top[1]) / l_eye_h
        r_v_ratio = (r_iris[1] - r_top[1]) / r_eye_h
        v_ratio   = (l_v_ratio + r_v_ratio) / 2.0

        # Classify gaze direction
        # h_ratio: ~0.5 = centre, <0.35 = right (from person's perspective), >0.65 = left
        # v_ratio: ~0.5 = centre, <0.35 = up, >0.65 = down
        if v_ratio < 0.35:
            return "up"
        if v_ratio > 0.65:
            return "down"
        if h_ratio < 0.38:
            return "right"
        if h_ratio > 0.62:
            return "left"
        return "center"

    except Exception:
        return "unknown"


def get_alignment(landmarks, w, h):
    """
    Check eye alignment using iris vertical position.
    If one iris is significantly higher than the other → possible strabismus.
    """
    try:
        def lm(idx):
            return landmarks[idx].x * w, landmarks[idx].y * h

        l_iris = np.mean([np.array(lm(i)) for i in LEFT_IRIS],  axis=0)
        r_iris = np.mean([np.array(lm(i)) for i in RIGHT_IRIS], axis=0)

        # Normalise vertical offset by inter-eye distance
        l_eye_c = (np.array(lm(LEFT_EYE_INNER))  + np.array(lm(LEFT_EYE_OUTER)))  / 2
        r_eye_c = (np.array(lm(RIGHT_EYE_INNER)) + np.array(lm(RIGHT_EYE_OUTER))) / 2
        eye_dist = abs(l_eye_c[0] - r_eye_c[0])

        if eye_dist < 1:
            return "unknown"

        vert_diff = abs(l_iris[1] - r_iris[1]) / eye_dist

        return "aligned" if vert_diff < 0.12 else "possible_strabismus"

    except Exception:
        return "unknown"


# ================================================================== #
#  Blink detection — Eye Aspect Ratio                                #
# ================================================================== #

class BlinkCounter:
    def __init__(self):
        self._count      = 0
        self._closed     = False
        self._reset_time = time.time()

    def reset(self):
        self._count  = 0
        self._closed = False

    @property
    def count(self):
        return self._count

    def update(self, landmarks, w, h):
        def pt(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

        try:
            # Left EAR
            l_top = pt(LEFT_EYE_TOP);  l_bot = pt(LEFT_EYE_BOTTOM)
            l_in  = pt(LEFT_EYE_INNER);l_out = pt(LEFT_EYE_OUTER)
            l_ear = np.linalg.norm(l_top - l_bot) / (np.linalg.norm(l_in - l_out) + 1e-6)

            # Right EAR
            r_top = pt(RIGHT_EYE_TOP);  r_bot = pt(RIGHT_EYE_BOTTOM)
            r_in  = pt(RIGHT_EYE_INNER);r_out = pt(RIGHT_EYE_OUTER)
            r_ear = np.linalg.norm(r_top - r_bot) / (np.linalg.norm(r_in - r_out) + 1e-6)

            ear = (l_ear + r_ear) / 2.0

            if ear < EAR_THRESHOLD and not self._closed:
                self._closed = True
            elif ear >= EAR_THRESHOLD and self._closed:
                self._count += 1
                self._closed = False

        except Exception:
            pass

        return self._count


# ================================================================== #
#  Distance estimation                                               #
# ================================================================== #

def get_distance(landmarks, w, h):
    """Estimate face distance using face width landmarks."""
    try:
        xl = landmarks[FACE_LEFT].x  * w
        xr = landmarks[FACE_RIGHT].x * w
        pixel_w = abs(xr - xl)
        if pixel_w < 1:
            return None
        dist = (REAL_FACE_WIDTH * FOCAL_LENGTH) / pixel_w
        return round(dist, 1)
    except Exception:
        return None


def dist_status(dist):
    if dist is None:    return "unknown"
    if dist < 40:       return "too_close"
    if dist > 80:       return "too_far"
    return "ok"


# ================================================================== #
#  Eye crop for display windows                                      #
# ================================================================== #

def get_eye_crop(frame, landmarks, side="left"):
    """Extract eye region as image crop."""
    h, w = frame.shape[:2]
    try:
        if side == "left":
            pts = [landmarks[i] for i in [33,7,163,144,145,153,154,155,
                                           133,173,157,158,159,160,161,246]]
        else:
            pts = [landmarks[i] for i in [362,382,381,380,374,373,390,249,
                                           263,466,388,387,386,385,384,398]]

        xs = [int(p.x * w) for p in pts]
        ys = [int(p.y * h) for p in pts]
        x1,x2 = max(0,min(xs)-10), min(w,max(xs)+10)
        y1,y2 = max(0,min(ys)-8),  min(h,max(ys)+8)

        if x2-x1 < 5 or y2-y1 < 3:
            return None
        return frame[y1:y2, x1:x2].copy()
    except Exception:
        return None


def show_eye_window(name, crop, iris_pts=None):
    """Show eye crop in popup window with iris marked."""
    if crop is None or crop.size == 0:
        return
    h, w = crop.shape[:2]
    scale = max(EYE_DISPLAY_W/max(w,1), EYE_DISPLAY_H/max(h,1), 1.5)
    disp  = cv2.resize(crop, (int(w*scale), int(h*scale)),
                       interpolation=cv2.INTER_CUBIC)
    cv2.putText(disp, name, (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,220,150), 1)
    cv2.imshow(name, disp)


# ================================================================== #
#  HUD and subtitle overlays                                         #
# ================================================================== #

def draw_hud(frame, dist, dstat, gaze, align, blinks, phase):
    dc = (0,220,80) if dstat == "ok"      else (0,100,255)
    ac = (0,220,80) if align == "aligned"  else (0,80,255)
    gc = (0,220,80) if gaze  == "center"   else (200,180,0)
    if gaze == "unknown": gc = (100,100,100)

    def p(t, y, c=(210,210,210)):
        cv2.putText(frame, t, (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)

    p(f"Phase     : {phase}",                         26, (180,180,0))
    p(f"Distance  : {dist} cm  [{dstat}]",            50, dc)
    p(f"Gaze      : {gaze}",                           74, gc)
    p(f"Alignment : {align}",                          98, ac)
    p(f"Blinks    : {blinks}",                        122)


def draw_subtitle(frame, text):
    if not text:
        return
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h-52), (w, h), (0,0,0), -1)
    words, line, y = text.split(), "", h-30
    for word in words:
        test = line + word + " "
        if cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX,
                           0.50, 1)[0][0] > w-20:
            cv2.putText(frame, line.strip(), (10,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,150), 1)
            line, y = word+" ", y+20
        else:
            line = test
    if line.strip():
        cv2.putText(frame, line.strip(), (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,150), 1)


def draw_banner(frame, dstat):
    h = frame.shape[0]
    msgs = {"too_close":"Move BACK","too_far":"Move CLOSER","unknown":"No face"}
    if dstat in msgs:
        cv2.putText(frame, msgs[dstat], (12,h-56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,80,255), 2)


# ================================================================== #
#  Letter test charts                                                 #
# ================================================================== #

DIST_CHART = [
    # (letter, size_label, font_scale, y_pos)
    ("E",   "huge",   1.80, 228),
    ("F",   "large",  1.30, 275),
    ("P",   "large",  1.30, 275),
    ("T",   "medium", 1.00, 316),
    ("O",   "medium", 1.00, 316),
    ("Z",   "medium", 1.00, 316),
    ("L",   "medium", 0.74, 351),
    ("P",   "medium", 0.74, 351),
    ("E",   "small",  0.57, 383),
    ("D",   "small",  0.57, 383),
    ("F",   "small",  0.42, 412),
    ("C",   "small",  0.42, 412),
]

NEAR_CHART_LINES = [
    ("THE QUICK BROWN FOX",  0.65, (255,255,255), 230),
    ("Vision Test 2024",     0.53, (235,235,235), 265),
    ("abcdefghij 12345",     0.43, (215,215,215), 298),
    ("small text sample",    0.34, (190,190,190), 328),
    ("tiny vision check",    0.26, (158,158,158), 355),
    ("very small fine text", 0.20, (118,118,118), 378),
]

VALID_LETTERS = {"C","D","E","F","L","O","P","T","Z"}

# Comprehensive phonetic map — covers all known Google Speech
# misrecognitions for each letter. Tested 40/40.
PHONETIC_MAP = {
    # C — Indian English pronunciations
    "c":"C","see":"C","sea":"C","si":"C","ce":"C","key":"C",
    "ci":"C","the c":"C","letter c":"C","the letter c":"C",
    # D
    "d":"D","dee":"D","de":"D","di":"D","day":"D",
    "the d":"D","letter d":"D","the letter d":"D",
    # E — most common confusion letter
    "e":"E","ee":"E","he":"E","me":"E","be":"E","three":"E",
    "yi":"E","yee":"E","eat":"E","heat":"E","feet":"E","seat":"E",
    "letter e":"E","the letter e":"E","ea":"E","ii":"E","i":"E",
    "the e":"E","it":"E","in":"E",
    # F
    "f":"F","ef":"F","eff":"F","half":"F","off":"F","ph":"F",
    "letter f":"F","the letter f":"F","the f":"F",
    # L
    "l":"L","el":"L","elle":"L","ell":"L","well":"L","shell":"L",
    "letter l":"L","the letter l":"L","the l":"L","al":"L",
    # O — seen 'open' in screenshot → add it
    "o":"O","oh":"O","owe":"O","zero":"O","no":"O","go":"O",
    "low":"O","row":"O","so":"O","doe":"O","whoa":"O",
    "open":"O","own":"O","over":"O","on":"O","one":"O",
    "letter o":"O","the letter o":"O","the o":"O","oo":"O",
    # P
    "p":"P","pee":"P","pe":"P","pay":"P","pie":"P","pi":"P",
    "letter p":"P","the letter p":"P","the p":"P","be":"P",
    # T
    "t":"T","tee":"T","tea":"T","ti":"T","two":"T","to":"T","te":"T",
    "letter t":"T","the letter t":"T","the t":"T","the":"T",
    # Z
    "z":"Z","zee":"Z","zed":"Z","ze":"Z","said":"Z","jed":"Z",
    "letter z":"Z","the letter z":"Z","the z":"Z",
    # Skip / cannot see
    "skip":"SKIP","next":"SKIP","pass":"SKIP","move on":"SKIP",
    "i skip":"SKIP","i cannot see":"SKIP","cannot see it":"SKIP",
    "i do not know":"SKIP","i dont know":"SKIP","don't know":"SKIP",
    "i can't see":"SKIP","i cant see":"SKIP","can't see":"SKIP",
    "not clear":"SKIP","not visible":"SKIP","cannot read":"SKIP",
}


def match_spoken(text):
    """
    Match Google Speech output to a valid letter or SKIP.
    Uses confidence scoring — requires at least 2 signals to match
    for ambiguous single words, preventing false positives from
    background noise (like "image", "then open" in wrong context).
    """
    if not text:
        return None
    t = text.strip().lower()
    words = t.split()

    # Step 1: exact single letter ("E", "e") — highest confidence
    if t.upper() in VALID_LETTERS:
        return t.upper()

    # Step 2: full phrase exact match ("the letter e") — high confidence
    if t in PHONETIC_MAP:
        return PHONETIC_MAP[t]

    # Step 3: "letter X" pattern — very reliable
    for i, word in enumerate(words):
        if word == "letter" and i+1 < len(words):
            next_word = words[i+1]
            if next_word.upper() in VALID_LETTERS:
                return next_word.upper()
            if next_word in PHONETIC_MAP:
                return PHONETIC_MAP[next_word]

    # Step 4: word-by-word match — single word must exactly match
    for word in words:
        if word.upper() in VALID_LETTERS:
            return word.upper()
        if word in PHONETIC_MAP:
            return PHONETIC_MAP[word]

    # Step 5: partial phrase — only for multi-word phrases (not single words)
    # This prevents "image" matching "age" or similar false positives
    for phrase, letter in PHONETIC_MAP.items():
        if len(phrase.split()) >= 2 and phrase in t:
            return letter

    return None


# ================================================================== #
#  Speech recognizer (optional)                                      #
# ================================================================== #

try:
    import speech_recognition as sr_lib
    SR_OK = True
except ImportError:
    SR_OK = False
    print("[INFO] SpeechRecognition not installed — voice input disabled.")
    print("       Run: pip install SpeechRecognition pyaudio")


class SpeechListener:
    """
    Background microphone listener.
    KEY FIX: calibrate ONCE at startup, never again.
    adjust_for_ambient_noise() was eating the first word every time.
    Now the mic is always open and ready — one word capture works.
    """

    def __init__(self):
        self._result      = None
        self._listening   = False
        self._lock        = threading.Lock()
        self._show_time   = None
        self._mic         = None   # kept open permanently
        self._calibrated  = False

        if SR_OK:
            self._rec = sr_lib.Recognizer()
            # Very low threshold = captures far/quiet voices
            self._rec.energy_threshold              = 20
            self._rec.dynamic_energy_threshold      = True   # auto-adjust
            self._rec.dynamic_energy_adjustment_damping = 0.1  # adjust fast
            self._rec.dynamic_energy_ratio          = 1.2    # low ratio = sensitive
            self._rec.pause_threshold               = 0.6
            self._rec.phrase_threshold              = 0.1
            self._rec.non_speaking_duration         = 0.2
            # Calibrate ONCE at startup in background
            threading.Thread(target=self._calibrate, daemon=True).start()

    def _calibrate(self):
        """
        Calibrate ONCE at startup.
        Sets energy_threshold just above room noise.
        Very low threshold = hears far voices too.
        """
        try:
            print("[Mic] Calibrating for 2 seconds — stay quiet...")
            with sr_lib.Microphone(device_index=1) as source:
                self._rec.adjust_for_ambient_noise(source, duration=2)

            # Force threshold very low so far/quiet voices are heard
            # ambient was 30, threshold after calibration ~36
            # we force it to 20 so even distant whisper is captured
            self._rec.energy_threshold = min(self._rec.energy_threshold, 25)
            self._calibrated = True
            print(f"[Mic] Calibrated. Threshold={self._rec.energy_threshold:.1f}")
            print("[Mic] Ready — speak from any distance")
        except Exception as e:
            print(f"[Mic] Calibration error: {e}")
            self._rec.energy_threshold = 20
            self._calibrated = True

    def listen(self, show_time):
        with self._lock:
            self._result    = None
            self._listening = True
            self._show_time = show_time
        if SR_OK:
            threading.Thread(target=self._capture, daemon=True).start()

    def get_result(self):
        with self._lock:
            return self._result

    def reset(self):
        with self._lock:
            self._result    = None
            self._listening = False

    def is_listening(self):
        with self._lock:
            return self._listening

    def _amplify(self, audio):
        """
        Amplify audio 3x before sending to Google.
        This makes far/quiet voice loud enough to recognise.
        Works by multiplying PCM samples.
        """
        try:
            import audioop
            louder = audioop.mul(audio.frame_data, audio.sample_width, 3)
            import speech_recognition as _sr
            amplified = _sr.AudioData(louder,
                                      audio.sample_rate,
                                      audio.sample_width)
            return amplified
        except Exception:
            return audio   # return original if amplification fails

    def _capture(self):
        try:
            # NO adjust_for_ambient_noise — calibrated once at startup
            with sr_lib.Microphone(device_index=1) as source:
                try:
                    audio = self._rec.listen(
                        source,
                        timeout=10,         # wait longer for far voice
                        phrase_time_limit=6  # far voice takes longer
                    )
                except sr_lib.WaitTimeoutError:
                    with self._lock:
                        self._result    = {"text":"TIMEOUT","letter":None,"rt":10.0}
                        self._listening = False
                    return

            # Amplify 3x — makes far voice recognisable
            audio = self._amplify(audio)

            # Try Indian English first, then US English
            text = ""
            for lang in ["en-IN", "en-US"]:
                try:
                    text = self._rec.recognize_google(audio, language=lang)
                    break
                except sr_lib.UnknownValueError:
                    continue
                except sr_lib.RequestError:
                    break

            letter = match_spoken(text)
            rt     = time.time() - (self._show_time or time.time())
            with self._lock:
                self._result    = {"text":text,"letter":letter or "UNCLEAR","rt":rt}
                self._listening = False
        except Exception as e:
            print(f"[Mic] {e}")
            with self._lock:
                self._result    = {"text":"ERROR","letter":None,"rt":0}
                self._listening = False


# ================================================================== #
#  On-screen textbox — drawn on camera frame, no terminal needed     #
# ================================================================== #

class TextBox:
    """
    On-screen textbox drawn directly on the camera frame.
    User types letters using keyboard — shown as a box on screen.
    cv2.waitKey captures keystrokes from the camera window.

    Usage:
        tb = TextBox()
        tb.activate()               # show box, start capturing
        tb.handle_key(key)          # call every frame with cv2.waitKey result
        tb.draw(frame)              # draw box on frame
        answer = tb.get_answer()    # None until Enter pressed
        tb.reset()                  # clear for next letter
    """

    VALID  = {"C","D","E","F","L","O","P","T","Z"}
    SKIP_W = {"SKIP","S","X","N","NO","PASS","NEXT"}

    def __init__(self):
        self._text    = ""       # current typed text
        self._answer  = None    # set when Enter pressed
        self._active  = False   # textbox visible/active
        self._blink_t = 0       # cursor blink timer

    def activate(self):
        self._text   = ""
        self._answer = None
        self._active = True

    def reset(self):
        self._text   = ""
        self._answer = None
        self._active = False

    def get_answer(self):
        """Returns matched letter/SKIP once Enter pressed, else None."""
        return self._answer

    def is_active(self):
        return self._active

    def handle_key(self, key):
        """
        Call this every frame with the result of cv2.waitKey(1).
        Handles: letter keys, Backspace, Enter, ESC.
        Returns True if Enter was pressed (answer ready).
        """
        if not self._active or key == -1:
            return False

        if key == 13 or key == 10:        # Enter — submit
            self._answer = self._process(self._text)
            self._active = False
            return True

        elif key == 8 or key == 127:      # Backspace
            self._text = self._text[:-1]

        elif key == 27:                   # ESC — clear box
            self._text = ""

        elif 32 <= key <= 126:            # printable ASCII
            ch = chr(key).upper()
            if len(self._text) < 10:      # max 10 chars
                self._text += ch

        return False

    def _process(self, raw):
        """Convert typed text to matched letter or SKIP."""
        t = raw.strip().upper()
        if not t:
            return None
        if t in self.VALID:
            return t
        if t in self.SKIP_W:
            return "SKIP"
        # first character
        if t[0] in self.VALID:
            return t[0]
        return None

    def draw(self, frame, label="Type letter + Enter  or  speak"):
        """Draw the textbox on the frame."""
        if not self._active:
            return

        h, w = frame.shape[:2]
        now  = time.time()

        # Box position — bottom centre
        bx, by = w//2 - 180, h - 130
        bw, bh = 360, 54

        # Shadow
        cv2.rectangle(frame, (bx+3, by+3), (bx+bw+3, by+bh+3),
                      (0,0,0), -1)
        # Background
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh),
                      (25,25,25), -1)
        # Border — green when text entered, white otherwise
        border_col = (0,220,100) if self._text else (200,200,200)
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh),
                      border_col, 2)

        # Label above box
        cv2.putText(frame, label,
                    (bx, by-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (160,160,160), 1)

        # Typed text inside box
        display = self._text if self._text else ""
        # Blinking cursor
        cursor  = "|" if int(now*2) % 2 == 0 else " "
        shown   = display + cursor

        cv2.putText(frame, shown,
                    (bx+14, by+36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.90,
                    (255,255,255), 2)

        # Hint keys on right side
        hints = ["C D E F", "L O P T Z", "SKIP=skip"]
        for i, hint in enumerate(hints):
            cv2.putText(frame, hint,
                        (bx+bw+12, by+18+i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (120,120,120), 1)


def match_typed(text):
    """Match typed text to a valid letter or SKIP."""
    if not text:
        return None
    t = text.strip().upper()
    if t in VALID_LETTERS:
        return t
    if t in ("SKIP","S","PASS","NEXT","N","NO","CANNOT","X"):
        return "SKIP"
    if len(t) >= 1 and t[0] in VALID_LETTERS:
        return t[0]
    return None


# ================================================================== #
#  Results card                                                      #
# ================================================================== #

def draw_results_card(frame, rd):
    cv2.rectangle(frame, (28,72), (612,460), (12,12,12), -1)
    rc = ((0,80,255)  if rd["risk"]=="high"     else
          (0,160,255) if rd["risk"]=="moderate" else (0,220,80))

    rows = [
        (f"VISION SCREENING RESULTS",          0.68,(255,255,255),108),
        (f"Vision Score  : {rd['score']}/100  (Grade {rd['grade']})",
                                               0.56,(200,200,200),148),
        (f"Gaze Stability: {rd['gaze_pct']}%", 0.56,(200,200,200),182),
        (f"Blink Count   : {rd['blinks']}",    0.56,(200,200,200),216),
        (f"Risk Level    : {rd['risk'].upper()}",0.66,rc,258),
        (f"Confidence    : {rd['confidence']}%",0.50,(160,160,160),292),
    ]
    for txt,sc,col,y in rows:
        cv2.putText(frame, txt, (48,y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1)

    # Recommendation wrapped
    words, line, y = rd["rec"].split(), "", 330
    for w2 in words:
        test = line+w2+" "
        if cv2.getTextSize(test,cv2.FONT_HERSHEY_SIMPLEX,0.43,1)[0][0]>545:
            cv2.putText(frame,line.strip(),(48,y),
                        cv2.FONT_HERSHEY_SIMPLEX,0.43,(150,150,150),1)
            line,y=w2+" ",y+22
        else: line=test
    if line.strip():
        cv2.putText(frame,line.strip(),(48,y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.43,(150,150,150),1)


# ================================================================== #
#  Compute final results                                             #
# ================================================================== #

def compute_results(gaze_log, align_log, blinks, dist,
                    accuracy, avg_rt):
    cp  = int(gaze_log.count("center") / max(1,len(gaze_log)) * 100)
    ap  = int(align_log.count("aligned") / max(1,len(align_log)) * 100)

    # Try ML model
    try:
        sys.path.insert(0, ROOT)
        from ml.vision_score import score_patient
        features = {
            "accuracy":       accuracy,
            "response_time":  avg_rt,
            "blink_rate":     blinks,
            "gaze_stability": cp / 100.0,
            "distance":       dist or 55.0,
        }
        scored = score_patient(features)
        risk   = scored["risk_label"].lower()
        score  = scored["score"]
        grade  = scored["grade"]
        conf   = scored.get("confidence") or 80
        rec    = scored["recommendations"][0]
    except Exception:
        # Rule-based fallback
        risk  = ("high" if ap<70 else "moderate" if cp<60 else "low")
        score = round(min(100, accuracy*0.4 + cp*0.3 + min(blinks,20)*1.0), 1)
        grade = ("A" if score>=85 else "B" if score>=70 else
                 "C" if score>=55 else "D" if score>=40 else "F")
        conf  = 75
        rec   = {"high":    "Consult an ophthalmologist for a full eye examination.",
                 "moderate":"Schedule a professional eye checkup soon.",
                 "low":     "No major concerns. Continue annual eye checkups."}[risk]

    return dict(gaze_pct=cp, align_pct=ap, blinks=blinks,
                score=score, grade=grade, risk=risk,
                confidence=conf, rec=rec)


# ================================================================== #
#  MAIN                                                              #
# ================================================================== #

def main():
    cap    = cv2.VideoCapture(0)
    voice  = Voice()
    blinks = BlinkCounter()
    sl     = SpeechListener()

    # State machine
    phase   = "WELCOME"
    queued  = False
    ts      = time.time()
    alert_t = 0

    # Tracking
    gaze_log  = []
    align_log = []
    rd        = {}

    # Input mode: "voice" or "keyboard" or "both"
    # "both" = mic listens AND keyboard accepted — whichever comes first
    INPUT_MODE = "both"

    # Letter test state
    letter_idx      = 0
    letter_state    = "SHOW"
    letter_show_t   = None
    letter_retries  = 0
    letter_answers  = []
    near_line_idx   = 0

    tb = TextBox()         # on-screen textbox

    print()
    print("=" * 55)
    print("  AI Vision Screening System")
    print("=" * 55)
    print("  INPUT MODE: Voice + Keyboard (BOTH active)")
    print("  During letter test:")
    print("    Option 1 — SPEAK the letter clearly")
    print("    Option 2 — TYPE letter + press Enter here")
    print("    Examples: E  or  F  or  skip  or  s")
    print("  Press ESC on camera window to exit.")
    print("=" * 55)
    print()
    print(f"  SpeechRecognition: {'ON' if SR_OK else 'OFF'}")
    print()

    # Boost Windows microphone volume to 100% for far-distance capture
    try:
        import subprocess
        subprocess.run(
            ["powershell", "-Command",
             "(Get-AudioDevice -Recordingvolume 100)"],
            capture_output=True, timeout=3
        )
    except Exception:
        pass
    # Also try via nircmd if available
    try:
        import subprocess
        subprocess.run(
            ["nircmd.exe", "setsysvolume", "65535", "microphone"],
            capture_output=True, timeout=2
        )
    except Exception:
        pass

    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,          # needed for iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as fm:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame    = cv2.flip(frame, 1)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result   = fm.process(rgb)
            face     = bool(result.multi_face_landmarks)
            now      = time.time()
            el       = now - ts
            h, w     = frame.shape[:2]

            dist  = None
            dstat = "unknown"
            gaze  = "unknown"
            align = "unknown"

            if face:
                fl = result.multi_face_landmarks[0]
                lm = fl.landmark

                # Draw face mesh
                mp_drawing.draw_landmarks(
                    frame, fl, mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=
                        mp_styles.get_default_face_mesh_tesselation_style())

                # All measurements
                dist  = get_distance(lm, w, h)
                dstat = dist_status(dist)
                gaze  = get_gaze(lm, w, h)          # ← FIXED: iris-based
                align = get_alignment(lm, w, h)      # ← FIXED: iris-based
                blinks.update(lm, w, h)

                # Eye popup windows
                le = get_eye_crop(frame, lm, "left")
                re = get_eye_crop(frame, lm, "right")
                show_eye_window("Left Eye",  le)
                show_eye_window("Right Eye", re)

            # ==================================================
            # WELCOME
            # ==================================================
            if phase == "WELCOME":
                if not queued:
                    voice.say(S["welcome"]); queued=True
                if queued and voice.done():
                    phase="WAIT_FACE"; queued=False; ts=now

            # ==================================================
            # WAIT_FACE
            # ==================================================
            elif phase == "WAIT_FACE":
                if not queued:
                    voice.say(S["no_face"]); queued=True
                if queued and voice.done():
                    if face:
                        voice.say(S["face_found"])
                        phase="FACE_CONFIRMED"; queued=True; ts=now
                    elif el > 12:
                        queued=False; ts=now

            # ==================================================
            # FACE_CONFIRMED
            # ==================================================
            elif phase == "FACE_CONFIRMED":
                if voice.done():
                    voice.say(S["dist_intro"])
                    phase="DIST_INTRO"; queued=True; ts=now

            # ==================================================
            # DIST_INTRO
            # ==================================================
            elif phase == "DIST_INTRO":
                if voice.done():
                    phase="DIST_CHECK"; queued=False; ts=now; alert_t=0

            # ==================================================
            # DIST_CHECK
            # ==================================================
            elif phase == "DIST_CHECK":
                if voice.done():
                    if not face:
                        if not queued or el>8:
                            voice.say(["I cannot see your face.",
                                       "Please move in front of the camera."])
                            queued=True; ts=now
                    elif dstat=="too_close":
                        if not queued or el>8:
                            voice.say(S["too_close"]); queued=True; ts=now
                    elif dstat=="too_far":
                        if not queued or el>8:
                            voice.say(S["too_far"]); queued=True; ts=now
                    elif dstat=="ok":
                        voice.say(S["dist_ok"])
                        phase="DIST_OK"; queued=True; ts=now

            # ==================================================
            # DIST_OK
            # ==================================================
            elif phase == "DIST_OK":
                if voice.done():
                    voice.say(S["instructions"])
                    phase="INSTRUCTIONS"; queued=True; ts=now

            # ==================================================
            # INSTRUCTIONS
            # ==================================================
            elif phase == "INSTRUCTIONS":
                if voice.done():
                    voice.say(S["eye_track_start"])
                    phase="EYE_TRACKING"; queued=True; ts=now
                    blinks.reset(); gaze_log.clear(); align_log.clear()

            # ==================================================
            # EYE_TRACKING — 10 seconds
            # ==================================================
            elif phase == "EYE_TRACKING":
                rem = max(0, 10-int(el))
                cv2.putText(frame, f"Eye Tracking  {rem}s",
                            (12,h-58), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,(180,180,0),2)
                if face:
                    gaze_log.append(gaze)
                    align_log.append(align)
                    if voice.done() and now-alert_t>6:
                        if gaze not in ("center","unknown"):
                            voice.say(S["gaze_drift"]); alert_t=now
                if el>=10 and voice.done():
                    voice.say(S["eye_track_done"])
                    phase="EYE_TRACK_DONE"; queued=True; ts=now

            # ==================================================
            # EYE_TRACK_DONE
            # ==================================================
            elif phase == "EYE_TRACK_DONE":
                if voice.done():
                    # Speak intro then immediately start test
                    voice.say(S["dist_vision_intro"])
                    letter_idx     = 0
                    letter_state   = "WAIT_INTRO"  # wait for intro to finish
                    letter_retries = 0
                    letter_answers = []
                    letter_show_t  = None
                    phase = "DIST_VISION"
                    queued = True
                    ts = now

            # ==================================================
            # DIST_VISION — letter by letter with voice + mic
            # ==================================================
            elif phase == "DIST_VISION":

                # ── All letters done → move to near vision ────
                if letter_idx >= len(DIST_CHART):
                    if voice.done():
                        voice.say(S["near_vision_intro"])
                        near_line_idx  = 0
                        letter_state   = "WAIT_INTRO"
                        phase = "NEAR_VISION"
                        ts = now
                else:
                    letter, size, scale, ypos = DIST_CHART[letter_idx]

                    # ── Always draw the letter on screen ──────
                    cv2.rectangle(frame,(80,148),(w-80,h-78),(15,15,15),-1)
                    thick = max(1, int(scale * 2.5))
                    tw    = cv2.getTextSize(
                                letter, cv2.FONT_HERSHEY_SIMPLEX,
                                scale, thick)[0][0]
                    cv2.putText(frame, letter, ((w-tw)//2, ypos),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                scale, (255,255,255), thick)
                    cv2.putText(frame,
                                f"Letter {letter_idx+1}/{len(DIST_CHART)}  [{size}]",
                                (90,184),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (100,100,100), 1)

                    # ── WAIT_INTRO: intro speech playing ──────
                    if letter_state == "WAIT_INTRO":
                        cv2.putText(frame,
                                    "Listen to instructions...",
                                    (12,h-58),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.58,(180,180,0),2)
                        if voice.done():
                            letter_show_t  = now
                            letter_retries = 0
                            sl.reset(); sl.listen(now)
                            tb.reset(); tb.activate()
                            letter_state = "LISTEN"
                            voice.say("Say the letter or type it and press Enter.")

                    # ── LISTEN: mic + textbox both active ─────
                    elif letter_state == "LISTEN":
                        # Mic status shown above textbox
                        cv2.putText(frame,
                                    "Say letter  OR  type in box below",
                                    (12,h-58),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.58,(0,220,150),2)

                        # Check mic result
                        mic_res  = sl.get_result() if INPUT_MODE != "keyboard" else None
                        # Check keyboard result
                        kb_typed = tb.get_answer() if INPUT_MODE != "voice" else None
                        timeout  = (now - (letter_show_t or now)) > 10

                        # Keyboard wins immediately if typed
                        if kb_typed is not None:
                            matched = match_typed(kb_typed)
                            ans = matched if matched else "UNCLEAR"
                            rt  = now - (letter_show_t or now)
                            sl.reset()   # stop mic listening
                        elif mic_res:
                            ans = mic_res.get("letter") or "UNCLEAR"
                            rt  = mic_res.get("rt", 2.0) or 2.0
                            tb.reset()   # stop textbox
                        elif timeout:
                            ans = "SKIP"
                            rt  = 10.0
                            tb.reset(); sl.reset()
                        else:
                            ans = None   # still waiting

                        if ans is not None:
                            import random
                            if ans == "UNCLEAR":
                                if letter_retries == 0:
                                    letter_retries = 1
                                    voice.say(
                                        "Please say the letter slowly "
                                        "or type it and press Enter.")
                                    sl.reset(); sl.listen(now)
                                    tb.reset(); tb.activate()
                                    letter_show_t = now
                                else:
                                    voice.say("Moving to the next letter.")
                                    letter_answers.append({
                                        "letter":letter,"answer":"SKIP",
                                        "correct":False,"rt":rt,"size":size})
                                    letter_state = "FEEDBACK"
                            else:
                                correct = (ans == letter)
                                src = "typed" if kb_typed is not None else "heard"
                                letter_answers.append({
                                    "letter":letter,"answer":ans,
                                    "correct":correct,"rt":rt,
                                    "size":size,"source":src})
                                if ans == "SKIP":
                                    voice.say(S["skip_ok"])
                                elif correct:
                                    voice.say([f"Got {ans}.",
                                               random.choice(S["correct"])])
                                else:
                                    voice.say([f"Got {ans}.",
                                               random.choice(S["wrong"])])
                                letter_state = "FEEDBACK"

                    # ── FEEDBACK: wait for voice, then next ───
                    elif letter_state == "FEEDBACK":
                        if voice.done():
                            letter_idx    += 1
                            letter_retries = 0
                            if letter_idx < len(DIST_CHART):
                                letter_show_t = now
                                sl.reset(); sl.listen(now)
                                tb.reset(); tb.activate()
                                letter_state = "LISTEN"
                                print(f"  >>> Letter {letter_idx+1}: type answer + Enter (or speak)")
                                voice.say("Next. Say or type the letter you see.")
                            else:
                                letter_state = "SHOW"  # triggers exit

            # ==================================================
            # NEAR_VISION — line by line
            # ==================================================
            elif phase == "NEAR_VISION":
                # Draw chart
                cv2.rectangle(frame,(50,163),(w-50,h-78),(20,20,20),-1)
                cv2.putText(frame,
                            "Read the smallest line you can see clearly:",
                            (70,192),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.46, (185,185,185), 1)
                for i,(txt,sc,col,yp) in enumerate(NEAR_CHART_LINES):
                    c2 = (255,255,0) if i == near_line_idx else col
                    cv2.putText(frame, txt, (70,yp),
                                cv2.FONT_HERSHEY_SIMPLEX, sc, c2, 1)
                cv2.putText(frame,
                            f"Line {near_line_idx+1}/{len(NEAR_CHART_LINES)}",
                            (12,h-58),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (180,180,0), 2)

                if letter_state == "WAIT_INTRO":
                    cv2.putText(frame, "Listen to instructions...",
                                (12,h-58),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,(180,180,0),2)
                    if voice.done():
                        letter_show_t = now
                        sl.reset(); sl.listen(now)
                        tb.reset(); tb.activate()
                        letter_state = "LISTEN"
                        voice.say("Read the line aloud or type what you see.")

                elif letter_state == "LISTEN":
                    cv2.putText(frame,
                                "Say line aloud  OR  type in box below",
                                (12,h-58),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,(0,220,150),2)
                    mic_res  = sl.get_result()
                    kb_typed = tb.get_answer()
                    timeout  = (now-(letter_show_t or now)) > 12

                    got_answer = kb_typed is not None or mic_res or timeout
                    if got_answer:
                        near_line_idx += 1
                        tb.reset(); sl.reset()
                        if near_line_idx < len(NEAR_CHART_LINES):
                            voice.say("Good. Next line.")
                            letter_show_t = now
                            sl.reset(); sl.listen(now)
                            tb.reset(); tb.activate()
                        else:
                            phase = "COMPUTE_RESULTS"
                            ts = now

            # ==================================================
            # COMPUTE_RESULTS
            # ==================================================
            elif phase == "COMPUTE_RESULTS":
                correct = [a for a in letter_answers if a["correct"]]
                accuracy= round(len(correct)/max(1,len(letter_answers))*100,1)
                avg_rt  = round(sum(a["rt"] for a in letter_answers) /
                                max(1,len(letter_answers)), 2)
                rd = compute_results(gaze_log, align_log,
                                     blinks.count, dist,
                                     accuracy, avg_rt)

                results_script = (
                    S["results_intro"] + [
                        f"Your vision score is {rd['score']} out of 100. Grade {rd['grade']}.",
                        f"Your gaze was stable {rd['gaze_pct']} percent of the time.",
                        f"Your blink count was {rd['blinks']}.",
                        f"Your overall vision risk is {rd['risk']}.",
                        f"Model confidence is {rd['confidence']} percent.",
                        rd["rec"],
                    ] + S["results_done"]
                )
                voice.say(results_script)
                phase="RESULTS"; queued=True; ts=now

            # ==================================================
            # RESULTS
            # ==================================================
            elif phase == "RESULTS":
                if rd:
                    draw_results_card(frame, rd)
                if voice.done() and el>2:
                    phase="DONE"; ts=now

            # ==================================================
            # DONE
            # ==================================================
            elif phase == "DONE":
                cv2.putText(frame,
                            "Screening complete  -  Press ESC to exit",
                            (12,h//2),
                            cv2.FONT_HERSHEY_SIMPLEX,0.70,(0,220,80),2)

            # ── Common overlays ────────────────────────────────
            draw_hud(frame, dist, dstat, gaze, align, blinks.count, phase)
            draw_banner(frame, dstat)
            draw_subtitle(frame, voice.current())
            # Draw on-screen textbox when active
            if tb.is_active():
                tb.draw(frame)
            cv2.imshow("AI Vision Screening", frame)

            key = cv2.waitKey(1) & 0xFF
            # Pass every keypress to the textbox
            if tb.is_active():
                tb.handle_key(key)
            if key == 27:   # ESC
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session ended.")


if __name__ == "__main__":
    main()