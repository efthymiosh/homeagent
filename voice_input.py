"""Voice input handling module.

Provides a simple interface to capture audio from the microphone and
return raw audio bytes for further processing (e.g., speech‑to‑text).
"""

import wave
from typing import Optional

import pyaudio


class VoiceInput:
    """Capture microphone audio.

    Usage::
        vi = VoiceInput()
        audio = vi.record(duration=5)
        # send `audio` to a STT engine
    """

    def __init__(self, device_index: Optional[int] = None, rate: int = 16000, chunk: int = 1024):
        self.rate = rate
        self.chunk = chunk
        self.device_index = device_index
        self._audio = pyaudio.PyAudio()

    def _open_stream(self):
        return self._audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk,
        )

    def record(self, duration: int = 5) -> bytes:
        """Record `duration` seconds of audio and return WAV bytes.

        Args:
            duration: Length of recording in seconds.
        Returns:
            Bytes containing a WAV file.
        """
        stream = self._open_stream()
        frames = []
        for _ in range(0, int(self.rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)
        stream.stop_stream()
        stream.close()

        # Write to an in‑memory WAV file
        import io
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self._audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(frames))
        return wav_io.getvalue()

    def close(self):
        """Terminate the underlying PyAudio instance."""
        self._audio.terminate()

# Simple demo when run directly
if __name__ == "__main__":
    vi = VoiceInput()
    print("Recording 3 seconds...")
    data = vi.record(3)
    with open("sample.wav", "wb") as f:
        f.write(data)
    print("Saved to sample.wav")
    vi.close()
