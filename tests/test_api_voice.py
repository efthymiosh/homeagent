"""Integration tests for the FastAPI voice endpoint."""

import io
import tempfile
import wave
import pytest
from fastapi.testclient import TestClient

from homeagent.api_voice import app

client = TestClient(app)

def _make_wav_bytes(duration_sec: float = 0.5, rate: int = 16000) -> bytes:
    """Create a short silent WAV file in memory for testing."""
    n_frames = int(duration_sec * rate)
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16‑bit
            wf.setframerate(rate)
            wf.writeframes(b"\x00\x00" * n_frames)
        return buf.getvalue()

def test_voice_endpoint_returns_transcript_and_response():
    wav_bytes = _make_wav_bytes()
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    response = client.post("/voice", files=files)
    assert response.status_code == 200
    data = response.json()
    # Whisper will return an empty string for pure silence, but the endpoint
    # should still include the keys.
    assert "transcript" in data
    assert "response" in data

def test_invalid_content_type():
    files = {"file": ("test.txt", b"not audio", "text/plain")}
    response = client.post("/voice", files=files)
    assert response.status_code == 400
    assert response.json()["detail"] == "Only WAV audio is supported"
