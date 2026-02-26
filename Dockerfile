FROM astral/uv:0.10.4 AS uv
FROM ubuntu:latest

COPY --from=uv /uv /bin/uv

# some more packages required here for kokoro and realtime tts
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    espeak-ng \
    portaudio19-dev \
    ca-certificates \
    alsa-base \
    alsa-utils \
    pipewire \
    libpipewire-0.3-dev \
    pipewire-audio-client-libraries \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app
RUN uv sync

RUN curl -L -o /app/voices-v1.0.bin \
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin && \
    curl -L -o /app/kokoro-v1.0.onnx \
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
    
COPY resources/system_prompt.md /app/system_prompt.md

RUN uv run python - <<EOF

import nltk
nltk.download('punkt_tab')

from faster_whisper import WhisperModel
WhisperModel("tiny.en", device="cpu", compute_type="int8")

EOF

ENV OPENAI_ENDPOINT="https://ai.efhd.dev/v1"
ENV OPENAI_MODEL="gpt-oss-120b"
ENV KOKORO_ONNX_PATH="/app/kokoro-v1.0.onnx"
ENV KOKORO_VOICES_PATH="/app/voices-v1.0.bin"

CMD ["uv", "run", "realtime_app.py"]
