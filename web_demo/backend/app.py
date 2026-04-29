import os
import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from inference import EmotionPredictor

app = FastAPI()
predictor = EmotionPredictor()

_FRONTEND = os.path.join(os.path.dirname(__file__), '..', 'frontend')

app.mount("/static", StaticFiles(directory=_FRONTEND), name="static")

#*
@app.get("/")
async def index():
    return FileResponse(os.path.join(_FRONTEND, 'index.html'))


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    sample_rate = 16000

    try:
        while True:
            message = await websocket.receive()

            if "text" in message and message["text"]:
                cfg = json.loads(message["text"])
                if cfg.get("type") == "config":
                    sample_rate = int(cfg.get("sample_rate", 16000))

            elif "bytes" in message and message["bytes"]:
                audio_np = np.frombuffer(message["bytes"], dtype=np.float32)
                if len(audio_np) < 512:
                    continue
                rms = float(np.sqrt(np.mean(audio_np ** 2)))
                if rms < 0.015:
                    await websocket.send_text(json.dumps({"silent": True}))
                    continue
                result = predictor.predict(audio_np, sample_rate)
                await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws] error: {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
