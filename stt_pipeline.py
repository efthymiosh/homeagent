"""Speech‑to‑text pipeline using Porcupine wake‑word detection and Whisper‑tiny.

* Porcupine continuously listens for a wake‑word (default: "hey jarvis").
* When the wake‑word is detected, we record a short audio snippet with
  :class:`VoiceInput` and transcribe it using ``whisper.load_model('tiny')``.
* The resulting text is returned for further processing.
"""

import threading
import time
from typing import Callable, Optional

import pvporcupine
import numpy as np
import whisper
import tempfile
import os

from voice_input import VoiceInput

class STTPipeline:
    """Manage wake‑word detection and transcription.

    Example usage::
        def handle(text: str):
            print("Heard:", text)
        pipeline = STTPipeline(on_transcript=handle)
        pipeline.start()
        # ... later
        pipeline.stop()
    """

    def __init__(self, on_transcript: Callable[[str], None],
                 wake_word: str = "hey jarvis",
                 record_seconds: int = 5,
                 porcupine_sensitivity: float = 0.6,
                 porcupine_access_key: str = "YOUR_ACCESS_KEY"):
        self.on_transcript = on_transcript
        self.wake_word = wake_word
        self.record_seconds = record_seconds
        self.porcupine_sensitivity = porcupine_sensitivity
        self.porcupine_access_key = porcupine_access_key
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # Load Whisper tiny model once
        self._model = whisper.load_model("tiny")

    def _detect_loop(self):
        # Initialize Porcupine with the custom wake‑word (built‑in keyword list)
        # Porcupine provides built‑in keywords; we map common names.
        keyword_paths = pvporcupine.KEYWORD_PATHS
        # Find a built‑in keyword that matches our phrase, fallback to "hey google"
        keyword = "hey jarvis" if "hey jarvis" in keyword_paths else "hey google"
        porcupine = pvporcupine.create(access_key=self.porcupine_access_key, keywords=[keyword], sensitivities=[self.porcupine_sensitivity])
        audio = VoiceInput(rate=porcupine.sample_rate, chunk=porcupine.frame_length)
        try:
            while self._running:
                pcm = audio._open_stream().read(porcupine.frame_length)
                # Convert bytes to int16 numpy array and then to list of ints for Porcupine
                pcm_int16 = np.frombuffer(pcm, dtype=np.int16)
                result = porcupine.process(pcm_int16.tolist())
                if result >= 0:
                    # Wake‑word detected
                    self._handle_wake_word()
        finally:
            audio.close()
            porcupine.delete()

    def _handle_wake_word(self):
        vi = VoiceInput(rate=16000)
        audio_bytes = vi.record(self.record_seconds)
        vi.close()
        # Whisper expects a file path or numpy array; write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            result = self._model.transcribe(tmp_path)
            text = result.get("text", "").strip()
            if text:
                self.on_transcript(text)
        finally:
            os.remove(tmp_path)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
            self._thread = None

# Simple demo when run directly
if __name__ == "__main__":
    def print_text(t: str):
        print("Transcribed:", t)
    pipeline = STTPipeline(on_transcript=print_text)
    print("Listening for wake‑word… (Ctrl+C to exit)")
    try:
        pipeline.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pipeline.stop()
        print("Stopped.")
