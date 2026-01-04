"""Speech‑to‑text pipeline using Porcupine wake‑word detection and Whisper‑tiny.

* Porcupine continuously listens for a wake‑word (default: "hey jarvis").
* When the wake‑word is detected, we record a short audio snippet with
  :class:`VoiceInput` and transcribe it using ``whisper.load_model('tiny')``.
* The resulting text is returned for further processing.
"""

import os
import tempfile
import threading
import time
from typing import Callable, Optional, Any

import whisper
from pocketsphinx import Decoder, get_model_path

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
                 kws_threshold: float = 1e-20):
        self.on_transcript = on_transcript
        self.wake_word = wake_word
        self.record_seconds = record_seconds
        self.kws_threshold = kws_threshold
        self._running = False
        self._thread: Optional[threading.Thread] = None
        model_path = get_model_path()
        # Placeholders for shared audio objects (created in start())
        self._audio: Optional[VoiceInput] = None
        self._stream: Optional[Any] = None
        # Initialize PocketSphinx Decoder directly with keyphrase mode
        self._decoder = Decoder(
            hmm=os.path.join(model_path, 'en-us/en-us'),
            dict=os.path.join(model_path, 'en-us/cmudict-en-us.dict'),
            keyphrase=self.wake_word,
            kws_threshold=self.kws_threshold,
        )
        # Load Whisper tiny model once
        self._model = whisper.load_model("tiny")

    def _detect_loop(self):
        # Use the shared VoiceInput stream created in start()
        audio_stream = self._stream
        try:
            while self._running:
                data = audio_stream.read(1024)
                if not data:
                    continue
                # Feed raw audio to decoder
                self._decoder.process_raw(data, False, False)
                hyp = self._decoder.hyp()
                if hyp is not None:
                    
                    # Wake‑word detected
                    self._handle_wake_word()
                    # Reset decoder for next detection
                    self._decoder.end_utt()
                    self._decoder.start_utt()
        finally:
            # Ensure utterance is ended before closing
            self._decoder.end_utt()
            # Stream will be closed in stop()


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
        # Create shared audio objects on the main thread
        self._audio = VoiceInput(rate=16000, chunk=1024)
        self._stream = self._audio._open_stream()
        # Start utterance processing for PocketSphinx
        self._decoder.start_utt()
        self._thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
            self._thread = None
        # Clean up audio resources
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._audio:
            self._audio.close()
            self._audio = None

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
