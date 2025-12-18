import asyncio
import base64
import contextlib
import logging
import re
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import WebSocket
from pydantic import BaseModel, Field
from starlette.datastructures import Address
from websockets import WebSocketException

RE_DATA_URL = re.compile(r"^data:(image/[^;]+)(;base64)?,(.+)$")

logger = logging.getLogger("pjm2jp.app.connection")


def fmtaddress(address: Address | None) -> str:
    return f"{address.host}:{address.port}" if address else "unknown"


class Status(BaseModel):
    address: Address | None = None
    connected: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class Frame(BaseModel):
    data: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    @property
    def nparray(self) -> np.ndarray:
        match = RE_DATA_URL.match(self.data)
        if not match:
            raise ValueError("Invalid data URL format")

        decoded = base64.b64decode(match.group(3))
        arr = np.frombuffer(decoded, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


class ProducerNotConnected(Exception):
    """Exception raised when a producer is not connected."""


class ProducerAlreadyConnected(Exception):
    """Exception raised when a producer is already connected."""


class Producer:
    def __init__(self) -> None:
        self.ws: WebSocket | None = None

    async def set(self, websocket: WebSocket) -> None:
        await websocket.accept()
        if self.ws is not None:
            raise ProducerAlreadyConnected("Producer already connected")
        self.ws = websocket
        logger.info("Producer connected (%s)", fmtaddress(websocket.client))

    def unset(self) -> None:
        if self.ws is not None:
            logger.info("Producer disconnected (%s)", fmtaddress(self.ws.client))
            self.ws = None

    async def receive(self) -> Frame:
        if self.ws is None:
            raise ProducerNotConnected("Producer not connected")
        data = await self.ws.receive_json()
        return Frame(**data)

    @property
    def status(self) -> Status:
        return Status(
            address=self.ws.client if self.ws else None,
            connected=bool(self.ws),
        )


class ConnectionManager:
    def __init__(self) -> None:
        self.producer = Producer()
        self.consumers: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        if not self.producer.ws:
            raise ProducerNotConnected("Producer not connected yet")
        self.consumers.add(websocket)
        logger.debug("Consumer connected (%s)", fmtaddress(websocket.client))

    def remove(self, websocket: WebSocket) -> None:
        self.consumers.discard(websocket)
        logger.debug("Consumer disconnected (%s)", fmtaddress(websocket.client))

    async def disconnect(
        self, websocket: WebSocket, code: int = 1000, reason: str = "bb"
    ) -> None:
        with contextlib.suppress(WebSocketException):
            await websocket.close(code=code, reason=reason)
        self.remove(websocket)

    async def broadcast(self, message: str) -> None:
        if not self.consumers:
            logging.warning("No consumers connected, skipping broadcast")
            return
        await asyncio.gather(
            *(consumer.send_text(message) for consumer in self.consumers)
        )
