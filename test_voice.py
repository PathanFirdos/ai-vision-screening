"""
Run this to confirm the fix works.
Command: python test_voice2.py
You should hear all 5 sentences.
"""
import pyttsx3
import time

sentences = [
    "Sentence one. Welcome to the vision screening system.",
    "Sentence two. Please sit comfortably in front of the camera.",
    "Sentence three. Make sure your face is clearly visible.",
    "Sentence four. The system will guide you through every step.",
    "Sentence five. Voice test is now complete.",
]

print("Testing fixed voice engine (fresh engine per sentence)...")

for i, s in enumerate(sentences):
    print(f"  Speaking: {s}")
    engine = pyttsx3.init()
    engine.setProperty('rate', 145)
    engine.setProperty('volume', 1.0)
    engine.say(s)
    engine.runAndWait()
    engine.stop()
    del engine
    print(f"  Done {i+1}")
    time.sleep(0.1)

print("All 5 sentences complete.")