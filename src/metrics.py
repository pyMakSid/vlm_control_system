from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class InferenceMetrics:

    ttft: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def tps(self) -> float:
        """Скорость генерации, токенов/с."""
        if self.generation_time <= 0 or self.completion_tokens <= 0:
            return 0.0
        return self.completion_tokens / self.generation_time


@dataclass
class MetricsTracker:
    """Скользящие средние по последним N запросам + расчёт FPS."""

    window: int = 30
    _frame_times: deque = field(default_factory=lambda: deque(maxlen=30))
    _ttft: deque = field(default_factory=lambda: deque(maxlen=30))
    _tps: deque = field(default_factory=lambda: deque(maxlen=30))
    _yolo_times: deque = field(default_factory=lambda: deque(maxlen=30))
    _total: int = 0

    def __post_init__(self) -> None:
        # Привязываем длину буферов к заданному окну.
        self._frame_times = deque(maxlen=self.window)
        self._ttft = deque(maxlen=self.window)
        self._tps = deque(maxlen=self.window)
        self._yolo_times = deque(maxlen=self.window)

    def update(self, m: InferenceMetrics, yolo_time: float | None = None) -> None:
        """Зарегистрировать метрики очередного обработанного кадра.

        yolo_time — время работы YOLO над кадром, с (если детекция включена).
        По нему считается FPS как пропускная способность детектора.
        """
        self._frame_times.append(perf_counter())
        self._ttft.append(m.ttft)
        self._tps.append(m.tps)
        if yolo_time is not None:
            self._yolo_times.append(yolo_time)
        self._total += 1

    @property
    def fps(self) -> float:
        """FPS по работе YOLO: 1 / среднее_время_детекции.

        Если YOLO выключен (времён нет) — откатываемся на сквозную частоту
        цикла по меткам времени.
        """
        if self._yolo_times:
            avg = self._avg(self._yolo_times)
            return 1.0 / avg if avg > 0 else 0.0
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        if span <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / span

    @staticmethod
    def _avg(values: deque) -> float:
        return sum(values) / len(values) if values else 0.0

    @property
    def avg_ttft(self) -> float:
        return self._avg(self._ttft)

    @property
    def avg_tps(self) -> float:
        return self._avg(self._tps)

    @property
    def total(self) -> int:
        return self._total

    def summary_line(self) -> str:
        """Однострочная сводка средних по окну."""
        return (
            f"FPS={self.fps:5.2f} | "
            f"TTFT={self.avg_ttft * 1000:6.0f}ms | "
            f"TPS={self.avg_tps:6.1f} | "
            f"кадров={self.total}"
        )
