import nltk
from typing import Generator
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.utils import convert_to_secret_str
from functools import partial
import os
from dotenv import load_dotenv
from RealtimeSTT import AudioToTextRecorder
import numpy as np
from kokoro_onnx import Kokoro
from kokoro_onnx.tokenizer import Tokenizer
import sounddevice as sd
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import time

# Load .env if present (optional)
load_dotenv()

# Configuration from environment
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT", "https://ai.efhd.dev/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-oss-120b")
KOKORO_ONNX_PATH = os.getenv("KOKORO_ONNX_PATH", "./resources/kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = os.getenv("KOKORO_VOICES_PATH", "./resources/voices-v1.0.bin")

tokenizer = Tokenizer()
kokoro = Kokoro(KOKORO_ONNX_PATH, KOKORO_VOICES_PATH)

llm = ChatOpenAI(
    model_name=OPENAI_MODEL,
    openai_api_base=OPENAI_ENDPOINT,
    openai_api_key=convert_to_secret_str(""),
    temperature=0.7,
    streaming=True,
    max_tokens=None,
)
agent = create_agent(
    llm,
    tools=[],
    checkpointer=InMemorySaver(),
)
config: RunnableConfig = {"configurable": {"thread_id": "1"}}


def speak_text(text: str):
    """Convert text to speech and play it via sounddevice."""

    for chunk in nltk.sent_tokenize(text):
        print(f"AI: {chunk}")
        phonemes = tokenizer.phonemize(chunk)

        voice = kokoro.get_voice_style("af_heart")
        samples, sample_rate = kokoro.create(
            phonemes, voice=voice, speed=1.0, is_phonemes=True
        )
        sd.wait()
        sd.play(samples, sample_rate)
    sd.wait()


def ask_openai(user_input: str) -> Generator[str, None, None]:
    """Send user_input to the LLM with memory and return response text."""

    for chunk in agent.stream({"messages": [HumanMessage(user_input)]}, config, stream_mode="updates"):
        yield str(chunk["model"]["messages"][-1].content)


def process_text(recorder: AudioToTextRecorder, text: str):
    """Callback for the realtime recorder – transcribe, query, speak.
    Pauses the recorder while speaking to avoid microphone feedback.
    """
    try:
        recorder.set_microphone(False)

        print(f"USER: {text}")
        for chunk in ask_openai(text):
            speak_text(chunk)

        recorder.set_microphone(True)
    except Exception as e:
        # Any error: speak a short notice and continue
        speak_text(f"Errored while processing text: {e}")


if __name__ == "__main__":
    recorder = AudioToTextRecorder(
        language="en",
        compute_type="int8",
        post_speech_silence_duration=0.2,
        enable_realtime_transcription=True,
        realtime_model_type="tiny.en",
        model="tiny.en",
        no_log_file=True,
        spinner=False,
    )
    print("Listening")
    while True:
        try:
            recorder.text(partial(process_text, recorder))
        except KeyboardInterrupt:
            print("\nInterrupted by user – exiting.")
            break
        except Exception as exc:
            # If the recorder itself fails, speak the error and keep looping
            speak_text(f"Recording failed with: {exc}")
            continue
    recorder.shutdown()
