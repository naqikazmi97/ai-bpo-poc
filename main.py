"""
main.py — FastAPI WebSocket entry point
Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
"""
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pipeline import VoicePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Voice Support Bot")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    log.info(f"[{session_id}] WebSocket connected")

    pipeline = VoicePipeline(session_id, websocket)
    session_ended = False

    try:
        while True:
            message = await websocket.receive()

            # Binary frame = raw PCM audio from client mic
            if "bytes" in message and message["bytes"]:
                await pipeline.feed_audio(message["bytes"])

            # Text frame = JSON control messages

            elif "text" in message and message["text"]:
                data = json.loads(message["text"])

                if data.get("type") == "end_of_audio":
                    await pipeline.end_audio()

                elif data.get("type") == "clear_session":
                    await pipeline.clear_session()

                elif data.get("type") == "bot_start":
                    await pipeline.bot_start()

                elif data.get("type") == "end_session":
                    # Explicit end — client pressed "end call"
                    # Triggers extraction before disconnect
                    log.info(f"[{session_id}] Explicit end_session received")
                    await pipeline.end_session()
                    session_ended = True
                    break

    except WebSocketDisconnect:
        log.info(f"[{session_id}] WebSocket disconnected")
    except Exception as e:
        log.error(f"[{session_id}] Unexpected error: {e}", exc_info=True)
    finally:
        # If client dropped without sending end_session,
        # cleanup() still runs extraction as a safety net
        await pipeline.cleanup(already_ended=session_ended)
