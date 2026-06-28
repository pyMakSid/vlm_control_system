from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np


class FrameGrabber:
    """Читает источник в фоновом потоке, хранит только последний кадр."""

    def __init__(self, source: str):
        self._src: object = int(source) if source.isdigit() else source
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._ended = False

    def start(self) -> FrameGrabber:
        self._cap = cv2.VideoCapture(self._src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Не удалось открыть источник '{self._src}'")
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                self._ended = True
                self._running = False
                break
            with self._lock:
                self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        """Вернуть копию последнего кадра (или None, если его ещё нет)."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    @property
    def ended(self) -> bool:
        """True, если источник закончился и новых кадров не будет."""
        return self._ended

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "FrameGrabber":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
