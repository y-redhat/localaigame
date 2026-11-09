from __future__ import annotations

import asyncio
import os
import secrets
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io

from oasis_runtime import OasisRuntime

ROOT = Path(__file__).resolve().parents[1]
OASIS_ROOT = Path(os.getenv("OASIS_ROOT", "/content/open-oasis"))
ACCESS_TOKEN = os.getenv("LOCALAI_ACCESS_TOKEN", "")
app = FastAPI(title="LocalAI Game")
runtime = None
runtime_lock = asyncio.Lock()
session_id = None


def check_token(value: str | None):
    if not ACCESS_TOKEN or not secrets.compare_digest(value or "", ACCESS_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid session token")


@app.get("/api/health")
async def health():
    return {"ok": True, "model_loaded": runtime is not None, "gpu": __import__("torch").cuda.is_available()}


@app.post("/api/session")
async def create_session(prompt: UploadFile = File(...), x_session_token: str | None = Header(default=None)):
    global runtime, session_id
    check_token(x_session_token)
    if prompt.content_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, or WebP image.")
    raw = await prompt.read()
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 12 MB or smaller.")
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(tmp.name, format="PNG")
    tmp.close()
    try:
        async with runtime_lock:
            runtime = await asyncio.to_thread(OasisRuntime, tmp.name, OASIS_ROOT)
            session_id = uuid.uuid4().hex
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {exc}")
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return {"session_id": session_id, "message": "Oasis session ready"}


@app.websocket("/ws/{requested_session}")
async def game_socket(websocket: WebSocket, requested_session: str, token: str = ""):
    global runtime, session_id
    if not ACCESS_TOKEN or not secrets.compare_digest(token, ACCESS_TOKEN):
        await websocket.close(code=4401)
        return
    if runtime is None or requested_session != session_id:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    try:
        while True:
            state = await websocket.receive_json()
            if not isinstance(state, dict):
                await websocket.send_json({"error": "Expected a JSON action object"})
                continue
            try:
                async with runtime_lock:
                    frame = await asyncio.to_thread(runtime.next_frame, state)
                buffer = io.BytesIO()
                frame.save(buffer, format="JPEG", quality=82)
                await websocket.send_bytes(buffer.getvalue())
            except Exception as exc:
                await websocket.send_json({"error": f"Frame generation failed: {exc}"})
                break
    except WebSocketDisconnect:
        pass


@app.get("/")
async def index():
    return FileResponse(ROOT / "frontend" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")
