from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]  # (x1, y1, x2, y2)


class YoloDetector:
    """Обёртка над ultralytics YOLO для предразметки кадра."""

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35):
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._conf = conf

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Прогоняет кадр через YOLO, возвращает список детекций."""
        results = self._model.predict(frame, conf=self._conf, verbose=False)
        detections: list[Detection] = []
        if not results:
            return detections

        r = results[0]
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            detections.append(
                Detection(
                    label=names.get(cls_id, str(cls_id)),
                    confidence=float(box.conf[0]),
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return detections

    @staticmethod
    def annotate(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.xyxy
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            caption = f"{d.label} {d.confidence:.2f}"
            cv2.putText(
                out, caption, (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )
        return out

    @staticmethod
    def summary(frame: np.ndarray, detections: list[Detection]) -> str:
        """Текстовая сводка детекций для промпта."""
        if not detections:
            return "YOLO: объектов не обнаружено."

        h, w = frame.shape[:2]
        parts = []
        for d in detections:
            x1, y1, x2, y2 = d.xyxy
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            horiz = "слева" if cx < w / 3 else "справа" if cx > 2 * w / 3 else "по центру"
            vert = "сверху" if cy < h / 3 else "снизу" if cy > 2 * h / 3 else "посередине"
            parts.append(f"{d.label} ({d.confidence:.2f}, {vert}-{horiz})")
        return "YOLO обнаружил: " + "; ".join(parts) + "."
