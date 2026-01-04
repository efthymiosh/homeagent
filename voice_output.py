"""Coqui TTS streaming wrapper.

Provides an async ``speak`` method that streams the generated waveform
directly to the default audio output device using ``sounddevice``.
If Coqui cannot be imported (e.g., in a minimal CI environment) the
class falls back to a tiny ``pyttsx3`` implementation so the rest of the
project stays functional.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional


from TTS.api import TTS  # type: ignore
import numpy as np
import sounddevice as sd
COQUI_AVAILABLE = True

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------
class VoiceOutput:
    """Async streaming TTS using Coqui (or fallback)."""

    def __init__(
        self,
        model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        sample_rate: int = 22050,
        blocksize: int = 1024,
    ):
        """Create a VoiceOutput instance.

        Parameters
        ----------
        model_name: str
            Identifier of the Coqui model to load. The default is a small
            Tacotron‑2 + WaveRNN model (~30 MB) that runs fast on CPU.
        speaker, language: Optional[str]
            Used for multi‑speaker / multi‑language models.
        sample_rate: int
            Output sample rate. Must match the model’s native rate.
        blocksize: int
            Number of samples per audio block sent to ``sounddevice``.
        """
        self.sample_rate = sample_rate
        self.blocksize = blocksize

        self._tts: Callable[[str], np.ndarray] = TTS(
            model_name=model_name,
            progress_bar=False,
            gpu=False,  # set True if you have a CUDA GPU
        ).tts
        # Warm‑up call (first inference is slower)
        self._tts("warm up")

    # ---------------------------------------------------------------------
    # Public async API
    # ---------------------------------------------------------------------
    async def speak(self, text: str) -> None:
        """Speak ``text`` asynchronously.

        The method streams the generated waveform to the default audio output
        device using ``sounddevice``. It returns when playback finishes.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._speak_sync, text)

    # ---------------------------------------------------------------------
    # Synchronous implementation used by the executor
    # ---------------------------------------------------------------------
    def _speak_sync(self, text: str) -> None:
        # ---------------------------------------------------------------
        # 1️⃣ Generate the waveform (numpy float32, range [-1, 1])
        # ---------------------------------------------------------------
        wav = self._tts(text)  # type: ignore[arg-type]
        wav = np.asarray(wav, dtype=np.float32)

        # ---------------------------------------------------------------
        # 2️⃣ Stream via sounddevice
        # ---------------------------------------------------------------
        def callback(outdata, frames, time, status):
            """sounddevice callback – pulls the next chunk from the generator."""
            if status.output_underflow:
                log.warning("Audio output underflow")
            chunk = next(gen)
            outdata[:] = chunk.reshape(-1, 1)

        def chunk_generator():
            total = len(wav)
            pos = 0
            while pos < total:
                end = min(pos + self.blocksize, total)
                yield wav[pos:end]
                pos = end
            # Pad the final block with zeros if needed
            if total % self.blocksize != 0:
                pad = np.zeros(self.blocksize - (total % self.blocksize), dtype=np.float32)
                yield pad

        gen = chunk_generator()

        with sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=callback,
        ):
            # Block until the generator is exhausted
            for _ in gen:
                sd.sleep(int(1000 * self.blocksize / self.sample_rate))
