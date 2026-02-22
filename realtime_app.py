from functools import partial
import os
import httpx
from dotenv import load_dotenv
from RealtimeSTT import AudioToTextRecorder
from kokoro import KPipeline
import sounddevice as sd

# Load .env if present (optional)
load_dotenv()

# Configuration from environment
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT", "https://ai.efhd.dev/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-oss-120b")

# Initialise TTS pipeline once
tts_pipeline = KPipeline(lang_code="a")


def speak_text(text: str):
    """Convert text to speech and play it via sounddevice."""
    generator = tts_pipeline(text, voice="af_heart")
    for _, _, audio in generator:
        sd.play(audio, 24000)
        sd.wait()


def speak_error(msg: str):
    """Speak a short error message."""
    speak_text(msg)


def ask_openai(user_input: str) -> str:
    """Send user_input to the OpenAI‑compatible endpoint and return the response text."""
    url = f"{OPENAI_ENDPOINT.rstrip('/')}/chat/completions"
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": user_input}],
        "temperature": 0.7,
    }
    try:
        resp = httpx.post(url, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"OpenAI request failed: {e}")


def process_text(recorder: AudioToTextRecorder, text: str):
    """Callback for the realtime recorder – transcribe, query, speak.
    Pauses the recorder while speaking to avoid microphone feedback.
    """
    try:
        recorder.set_microphone(False)

        reply = ask_openai(text)
        speak_text(reply)

        recorder.set_microphone(True)
    except Exception as e:
        # Any error: speak a short notice and continue
        speak_error(str(e))


if __name__ == "__main__":
    # Use the same recorder settings as the original example
    with AudioToTextRecorder(
        language="en",
        compute_type="int8",
        post_speech_silence_duration=0.2,
        enable_realtime_transcription=True,
        realtime_model_type="tiny.en",
        model="tiny.en",
        no_log_file=True,
    ) as recorder:
        while True:
            try:
                recorder.text(partial(process_text, recorder))
            except KeyboardInterrupt:
                print("\nInterrupted by user – exiting.")
                break
            except Exception as exc:
                # If the recorder itself fails, speak the error and keep looping
                speak_error(f"Failed with: {exc}")
                continue
