"""
speech_recognizer.py
--------------------
Captures child's spoken answer using microphone.
Runs in a background thread — never blocks the camera loop.

Library: SpeechRecognition  (pip install SpeechRecognition pyaudio)
Engine : Google Speech API (free, no key needed for basic use)
         Falls back to offline Vosk if no internet.

Usage:
    from app.vision.speech_recognizer import SpeechRecognizer

    sr = SpeechRecognizer()
    sr.listen()               # start listening in background
    answer = sr.get_result()  # None if not spoken yet, str if spoken
    sr.reset()                # ready for next letter
"""

import threading
import time
import queue

try:
    import speech_recognition as sr_lib
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("[WARNING] SpeechRecognition not installed.")
    print("          Run: pip install SpeechRecognition pyaudio")


# Letters the child is expected to say
VALID_LETTERS = {"C","D","E","F","L","O","P","T","Z"}

# Common misrecognitions → correct letter
PHONETIC_MAP = {
    # letter name spoken
    "see":   "C", "sea":  "C",
    "dee":   "D", "the":  "D",
    "ee":    "E", "he":   "E", "me": "E",
    "ef":    "F", "eff":  "F",
    "el":    "L", "elle": "L",
    "oh":    "O", "owe":  "O", "zero": "O",
    "pee":   "P", "pe":   "P",
    "tee":   "T", "tea":  "T",
    "zee":   "Z", "zed":  "Z", "sed": "Z",
    # single char spoken directly
    "c":"C","d":"D","e":"E","f":"F",
    "l":"L","o":"O","p":"P","t":"T","z":"Z",
    # skip commands
    "skip":"SKIP","next":"SKIP","pass":"SKIP",
    "i cannot see it":"SKIP",
    "cannot see":"SKIP",
    "don't know":"SKIP",
}


def match_letter(spoken_text: str) -> str | None:
    """
    Match a raw spoken string to a valid letter or SKIP.
    Returns letter string or None if no match.
    """
    if not spoken_text:
        return None

    text = spoken_text.strip().lower()

    # Direct single-char match
    if text.upper() in VALID_LETTERS:
        return text.upper()

    # Phonetic map
    if text in PHONETIC_MAP:
        return PHONETIC_MAP[text]

    # Check if any word in the phrase matches
    for word in text.split():
        if word.upper() in VALID_LETTERS:
            return word.upper()
        if word in PHONETIC_MAP:
            return PHONETIC_MAP[word]

    return None


class SpeechRecognizer:
    """
    Background microphone listener.
    The camera loop calls listen() when a letter appears,
    then polls get_result() until child speaks.
    """

    def __init__(self, timeout: float = 6.0, phrase_limit: float = 3.0):
        """
        timeout    : max seconds to wait for speech to start
        phrase_limit: max seconds to record one answer
        """
        self._timeout     = timeout
        self._phrase_limit = phrase_limit
        self._result_q    = queue.Queue()
        self._listening   = False
        self._lock        = threading.Lock()
        self._raw_text    = None   # last raw recognition result
        self._letter      = None   # matched letter
        self._listen_time = None   # when listen() was called
        self._answer_time = None   # when answer was received

        if not SR_AVAILABLE:
            return

        self._recognizer = sr_lib.Recognizer()
        self._recognizer.energy_threshold        = 300
        self._recognizer.dynamic_energy_threshold = True
        self._recognizer.pause_threshold          = 0.6

    # ---------------------------------------------------------------- #

    def listen(self):
        """
        Start listening for the child's answer in a background thread.
        Call this when a letter is displayed on screen.
        """
        with self._lock:
            if self._listening:
                return
            self._listening   = True
            self._raw_text    = None
            self._letter      = None
            self._listen_time = time.time()
            self._answer_time = None

        if not SR_AVAILABLE:
            # Simulation mode — auto-answer after 1.5s for testing
            threading.Thread(target=self._simulate, daemon=True).start()
            return

        threading.Thread(target=self._capture, daemon=True).start()

    def get_result(self) -> dict | None:
        """
        Returns None if child hasn't spoken yet.
        Returns dict once answer received:
            {
                "letter":        "E",           # matched letter or SKIP
                "raw":           "ee",          # what was heard
                "response_time": 1.34,          # seconds from listen() call
                "valid":         True           # False if no match
            }
        """
        with self._lock:
            if self._letter is None:
                return None
            return {
                "letter":        self._letter,
                "raw":           self._raw_text or "",
                "response_time": self._answer_time - self._listen_time
                                 if self._answer_time and self._listen_time
                                 else None,
                "valid":         self._letter is not None,
            }

    def reset(self):
        """Call this before displaying the next letter."""
        with self._lock:
            self._listening   = False
            self._raw_text    = None
            self._letter      = None
            self._listen_time = None
            self._answer_time = None

    def is_listening(self) -> bool:
        with self._lock:
            return self._listening

    # ---------------------------------------------------------------- #
    #  Internal                                                         #
    # ---------------------------------------------------------------- #

    def _capture(self):
        """Background thread: captures one spoken answer."""
        try:
            with sr_lib.Microphone() as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.3)
                try:
                    audio = self._recognizer.listen(
                        source,
                        timeout=self._timeout,
                        phrase_time_limit=self._phrase_limit
                    )
                except sr_lib.WaitTimeoutError:
                    self._set_result("TIMEOUT", None)
                    return

            # Try Google first
            try:
                text = self._recognizer.recognize_google(audio, language="en-US")
            except sr_lib.UnknownValueError:
                text = ""
            except sr_lib.RequestError:
                # No internet — try offline
                text = self._offline_fallback(audio)

            letter = match_letter(text)
            self._set_result(text, letter)

        except Exception as e:
            print(f"[SpeechRecognizer] Error: {e}")
            self._set_result("ERROR", None)

    def _offline_fallback(self, audio) -> str:
        """Try Vosk offline engine if available."""
        try:
            import vosk  # pip install vosk
            # Requires vosk model downloaded separately
            return ""
        except ImportError:
            return ""

    def _set_result(self, raw: str, letter: str | None):
        with self._lock:
            self._raw_text    = raw
            self._letter      = letter if letter else "UNCLEAR"
            self._answer_time = time.time()
            self._listening   = False

    def _simulate(self):
        """Simulation mode when SpeechRecognition not installed."""
        import random
        time.sleep(random.uniform(1.0, 2.5))
        letters = list(VALID_LETTERS)
        fake_letter = random.choice(letters)
        self._set_result(fake_letter.lower(), fake_letter)