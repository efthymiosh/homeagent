"""Tests for voice_input module."""
from voice_input import VoiceInput

import io
import wave
import numpy as np



def test_record_returns_wav_bytes(monkeypatch):
    """Ensure `record` returns valid WAV bytes.
    We mock PyAudio to avoid real microphone access.
    """
    class DummyStream:
        def __init__(self, frames):
            self.frames = frames
            self.idx = 0
        def read(self, chunk):
            # return chunk of zeros (16‑bit little endian)
            return (b"\x00\x00" * chunk)
        def stop_stream(self):
            pass
        def close(self):
            pass

    class DummyAudio:
        def __init__(self):
            pass
        def open(self, **kwargs):
            return DummyStream([])
        def get_sample_size(self, fmt):
            return 2
        def terminate(self):
            pass

    monkeypatch.setattr('homeagent.voice_input.pyaudio.PyAudio', DummyAudio)
    vi = VoiceInput(rate=16000, chunk=1024)
    wav_bytes = vi.record(duration=1)
    # Verify it's a valid WAV header
    with io.BytesIO(wav_bytes) as f:
        with wave.open(f, 'rb') as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
    vi.close()
