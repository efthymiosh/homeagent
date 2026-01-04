"""API for receiving voice audio from remote devices.

Provides a FastAPI endpoint that accepts raw WAV bytes (or multipart file)
and forwards them to the speech‑to‑text pipeline.
"""

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from router import CommandRouter
from voice_output import VoiceOutput
from stt_pipeline import STTPipeline

# Global pipeline instance
pipeline: STTPipeline | None = None

def _init_pipeline():
    global pipeline
    if pipeline is None:
        # Simple callback that stores the latest transcript
        async def _callback(text: str):
            # Store in app state for retrieval
            app.state.last_transcript = text
        # Wrap sync callback for async use
        def sync_cb(text: str):
            asyncio.create_task(_callback(text))
        pipeline = STTPipeline(on_transcript=sync_cb)
        pipeline.start()

@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_pipeline()
    app.state.last_transcript = ""
    try:
        yield
    finally:
        if pipeline:
            pipeline.stop()

app = FastAPI(title="HomeAgent Voice API", lifespan=lifespan)

# Initialise router and TTS (singletons)
router = CommandRouter()
tts = VoiceOutput()

@app.post("/voice")
async def receive_voice(file: UploadFile = File(...)) -> Any:
    """Receive a WAV audio file, transcribe it, route to a handler, and return the result.
    """
    if file.content_type not in ("audio/wav", "audio/x-wav"):
        raise HTTPException(status_code=400, detail="Only WAV audio is supported")
    content = await file.read()
    # Direct transcription using Whisper tiny (fast)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(tmp_path, fp16=False)
        text = result.get("text", "").strip()
    finally:
        os.remove(tmp_path)
    # Store raw transcript
    app.state.last_transcript = text
    # Route to a handler (may return a response string)
    response = router.route(text)
    return JSONResponse(content={"transcript": text, "response": response or ""})

@app.get("/last_transcript")
async def get_last_transcript():
    return JSONResponse(content={"last_transcript": getattr(app.state, "last_transcript", "")})

# If running directly, start the server (use uvicorn)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
