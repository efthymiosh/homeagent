"""API for receiving voice audio from remote devices.

Provides a FastAPI endpoint that accepts raw WAV bytes (or multipart file)
and forwards them to the speech‑to‑text pipeline.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Any
import asyncio
import tempfile, os

from stt_pipeline import STTPipeline

app = FastAPI(title="HomeAgent Voice API")

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

@app.on_event("startup")
async def startup_event():
    _init_pipeline()
    app.state.last_transcript = ""

@app.on_event("shutdown")
async def shutdown_event():
    if pipeline:
        pipeline.stop()

@app.post("/voice")
async def receive_voice(file: UploadFile = File(...)) -> Any:
    """Receive a WAV audio file from a network device and transcribe it.
    """
    if file.content_type not in ("audio/wav", "audio/x-wav"):
        raise HTTPException(status_code=400, detail="Only WAV audio is supported")
    content = await file.read()
    # Directly transcribe using Whisper tiny (quick path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        import whisper
        model = whisper.load_model("tiny")
        result = model.transcribe(tmp_path)
        text = result.get("text", "").strip()
    finally:
        os.remove(tmp_path)
    # Store latest transcript
    app.state.last_transcript = text
    return JSONResponse(content={"transcript": text})

@app.get("/last_transcript")
async def get_last_transcript():
    return JSONResponse(content={"last_transcript": getattr(app.state, "last_transcript", "")})

# If running directly, start the server (use uvicorn)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
