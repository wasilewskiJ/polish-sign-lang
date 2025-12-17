import logging
import queue
import threading
from collections import deque
from pathlib import Path

from pydantic import BaseModel
from translator.tf import PJMClassifier

from app.connection import Frame

logger = logging.getLogger("pjm2jp.app.evaluator")


class EvaluationResult(BaseModel):
    frame: Frame
    letter: str
    ensured: bool = False


class Evaluator:
    def __init__(self, consistency: int = 5) -> None:
        self._classifier: PJMClassifier | None = None
        self._classification_buffer: deque[str] = deque(maxlen=consistency)
        self._queue = queue.Queue(maxsize=1)
        self._result = None
        self._thread = threading.Thread(target=self._process_frame, daemon=True)
        self._thread.start()

    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Classifier model not found at {path}")
        logger.info("Loading classifier from %s", path)
        self._classifier = PJMClassifier(str(path))

    def evaluate(self, frame: Frame) -> EvaluationResult:
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self._queue.get_nowait()
            self._queue.put_nowait(frame)
        return self._result or EvaluationResult(frame=frame, letter="", ensured=False)

    def _process_frame(self) -> None:
        while True:
            frame = self._queue.get()
            letter, _ = self._classifier.process_frame(frame.nparray)
            if letter is not None:
                self._classification_buffer.append(letter)
            self._result = EvaluationResult(
                frame=frame,
                letter=letter or "",
                ensured=(
                    len(self._classification_buffer) == self._classification_buffer.maxlen
                    and all(item == letter for item in self._classification_buffer)
                ),
            )

    def reset(self) -> None:
        self._classification_buffer.clear()
