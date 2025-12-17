import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.connection import ConnectionManager, ProducerAlreadyConnected, ProducerNotConnected, Status
from app.evaluator import Evaluator

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pjm2jp.app")

manager: ConnectionManager = ConnectionManager()
evaluator = Evaluator(consistency=10)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    path = Path(__file__).parent.parent / "translator" / "models" / "pjm_model.keras"
    evaluator.load(path=path.absolute())
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/stream-status")
async def status() -> Status:
    return manager.producer.status


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket) -> None:
    try:
        await manager.producer.set(websocket)
    except ProducerAlreadyConnected:
        await websocket.close(code=1013, reason="Another producer already connected")

    try:
        while True:
            frame = await manager.producer.receive()
            if manager.consumers:
                evaluation = evaluator.evaluate(frame)
                message = evaluation.model_dump_json()
                await manager.broadcast(message)
            await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        manager.producer.unset()
        await asyncio.gather(
            *[
                manager.disconnect(consumer, code=1013, reason="Producer disconnected")
                for consumer in manager.consumers
            ],
            return_exceptions=True,
        )


@app.websocket("/ws/join")
async def join(websocket: WebSocket) -> None:
    try:
        await manager.connect(websocket)
    except ProducerNotConnected:
        await websocket.close(code=1013, reason="Producer not connected yet")

    try:
        while True:
            await websocket.receive_text()
            await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        manager.remove(websocket)


def main() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
