from typing import Generator
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.utils import convert_to_secret_str
from functools import partial
import os
from dotenv import load_dotenv
from RealtimeSTT import AudioToTextRecorder
from kokoro import KPipeline
import sounddevice as sd
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

# Load .env if present (optional)
load_dotenv()

# Configuration from environment
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT", "https://ai.efhd.dev/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-oss-120b")

# Initialise TTS pipeline once
tts_pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", trf=False)


def speak_text(text: str):
    """Convert text to speech and play it via sounddevice."""
    generator = tts_pipeline(text, voice="af_heart,af_bella")
    for _, _, audio in generator:
        sd.play(audio, 24000)
        sd.wait()


def speak_error(msg: str):
    """Speak a short error message."""
    speak_text(msg)


# Initialise LLM with endpoint and model via OpenAI compatible API
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

def ask_openai(user_input: str) -> Generator[str, None, None]:
    """Send user_input to the LLM with memory and return response text."""

    for chunk in agent.stream(
        {"messages": [HumanMessage(user_input)]},
        config,
        stream_mode="updates"
    ):
        yield str(chunk["model"]["messages"][-1].content)


def process_text(recorder: AudioToTextRecorder, text: str):
    """Callback for the realtime recorder – transcribe, query, speak.
    Pauses the recorder while speaking to avoid microphone feedback.
    """
    try:
        recorder.set_microphone(False)

        print(f"USER: {text}")
        for chunk in ask_openai(text):
            print(f"AI: {chunk}")
            speak_text(chunk)

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
        spinner=False,
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
