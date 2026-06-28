"""Клиент для маловесной VLM, запущенной на сервере llama.cpp.

llama.cpp server (`llama-server`) предоставляет OpenAI-совместимый эндпоинт
`/v1/chat/completions`. Мультимодальные модели (LLaVA, Qwen2-VL, MiniCPM-V,
SmolVLM и т.п.) принимают изображение как data-URL в поле image_url.

Запуск сервера, пример:
    llama-server -m model.gguf --mmproj mmproj.gguf --port 8080

Клиент держит скользящее окно из последних N кадров вместе с командами,
которые модель по ним выдала. Это даёт VLM временной контекст: она видит
движение сцены и собственные прошлые решения, а не только текущий кадр.
"""

from __future__ import annotations

import base64
import json
from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import cv2
import numpy as np
import requests

from .metrics import InferenceMetrics


@dataclass
class VLMConfig:
    base_url: str = "http://127.0.0.1:8080"
    model: str = "local-vlm"          # для llama.cpp имя модели не критично
    temperature: float = 0.1          # низкая — нам нужны детерминированные команды
    max_tokens: int = 256
    timeout: float = 30.0             # секунды на запрос
    jpeg_quality: int = 80            # качество кодирования кадра
    max_image_side: int = 512         # ресайз длинной стороны кадра (0 = не менять)
    context_frames: int = 3           # сколько кадров держать в окне (1 = stateless)
    # Передавать ли в контекст прошлые ОТВЕТЫ модели. По умолчанию False:
    # в окно идут только прошлые кадры (оценка движения), без ответов —
    # это исключает самоподкрепление (дословное копирование команды) у
    # маловесных VLM. True воспроизводит прежнее поведение.
    replay_decisions: bool = False
    # Путь чат-эндпоинта на сервере. llama.cpp: /v1/chat/completions;
    # RKLLM flask_server из rknn-llm: обычно /rkllm_chat (свериться с репо).
    endpoint_path: str = "/v1/chat/completions"
    # Путь проверки готовности. У RKLLM-демо /health может не быть —
    # пустая строка отключает проверку.
    health_path: str = "/health"
    # JSON-схема ответа для constrained-декодирования (llama.cpp поддерживает
    # response_format=json_schema). None — без ограничения. Заставляет модель
    # вернуть валидный JSON со строго допустимой командой.
    response_schema: Optional[dict] = None
    # Отключить режим «мышления» у reasoning-моделей (Qwen3 и т.п.). Иначе
    # модель тратит весь бюджет токенов на скрытое рассуждение и не отдаёт
    # ответ. True добавляет chat_template_kwargs.enable_thinking=false.
    disable_thinking: bool = True


class VLMClient:
    """Тонкая обёртка над chat/completions API сервера (llama.cpp / RKLLM).

    Вся логика (streaming, base64-кадр, окно контекста, метрики) общая;
    разные бэкенды отличаются лишь base_url и endpoint_path в конфиге.
    Если схема запроса/ответа у RKLLM-сервера расходится с OpenAI —
    правка локализована в _stream()/_build_messages().
    """

    def __init__(self, config: Optional[VLMConfig] = None):
        self.config = config or VLMConfig()
        self._endpoint = (
            self.config.base_url.rstrip("/") + self.config.endpoint_path
        )
        self._session = requests.Session()
        # Храним предыдущие (context_frames - 1) кадров и ответов модели по ним.
        # Текущий кадр в окно не входит, поэтому maxlen на единицу меньше.
        history_len = max(0, self.config.context_frames - 1)
        self._history: deque[dict] = deque(maxlen=history_len)

    def reset(self) -> None:
        """Очистить окно контекста (например, при смене миссии)."""
        self._history.clear()

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """Ужимает кадр до max_image_side по длинной стороне — меньше токенов."""
        limit = self.config.max_image_side
        if not limit:
            return frame
        h, w = frame.shape[:2]
        side = max(h, w)
        if side <= limit:
            return frame
        scale = limit / side
        return cv2.resize(
            frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

    def _encode_frame(self, frame: np.ndarray) -> str:
        """Кодирует кадр OpenCV (BGR ndarray) в data-URL base64 JPEG."""
        frame = self._resize(frame)
        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
        )
        if not ok:
            raise ValueError("Не удалось закодировать кадр в JPEG")
        b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _build_messages(
        self, data_url: str, system_prompt: str, mission_prompt: str
    ) -> list[dict]:
        """Собирает messages: system + история кадров/команд + текущий кадр."""
        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        # Прошлые кадры (от старого к свежему). Ответы модели подмешиваем в
        # контекст только при replay_decisions=True (иначе — лишь кадры, чтобы
        # модель оценивала движение, но не копировала свою прошлую команду).
        history = list(self._history)
        n = len(history)
        for i, item in enumerate(history):
            age = n - i  # на сколько шагов назад (t-1, t-2, ...)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Предыдущий кадр (t-{age})."},
                        {"type": "image_url", "image_url": {"url": item["data_url"]}},
                    ],
                }
            )
            if self.config.replay_decisions and item.get("response"):
                messages.append({"role": "assistant", "content": item["response"]})

        # Текущий кадр + миссия — здесь модель должна выдать команду.
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{mission_prompt}\nТекущий кадр (t). Дай команду.",
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        )
        return messages

    def infer(self, frame: np.ndarray, system_prompt: str, mission_prompt: str) -> dict:
        """Отправляет кадр (с учётом окна контекста) и два промпта в VLM.

        Запрос идёт в streaming-режиме — это позволяет замерить TTFT (момент
        прихода первого токена) и TPS.

        Возвращает dict вида:
            {"command": str, "reason": str, "confidence": float, "raw": str,
             "metrics": InferenceMetrics}
        """
        data_url = self._encode_frame(frame)
        messages = self._build_messages(data_url, system_prompt, mission_prompt)

        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # Ограничиваем вывод JSON-схемой (если задана и поддерживается сервером).
        if self.config.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "drone_command",
                    "strict": True,
                    "schema": self.config.response_schema,
                },
            }
        # Отключаем «мышление» reasoning-моделей (Qwen3 и т.п.), чтобы они не
        # тратили весь бюджет токенов на скрытое рассуждение вместо ответа.
        if self.config.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload["reasoning_budget"] = 0

        content, metrics = self._stream(payload)
        result = self._parse_response(content)
        result["metrics"] = metrics

        # Кладём кадр в окно контекста; ответ модели сохраняем только если он
        # будет воспроизводиться (replay_decisions), иначе храним лишь кадр.
        if self._history.maxlen:
            compact = None
            if self.config.replay_decisions:
                compact = json.dumps(
                    {
                        "command": result["command"],
                        "reason": result["reason"],
                        "confidence": result["confidence"],
                    },
                    ensure_ascii=False,
                )
            self._history.append({"data_url": data_url, "response": compact})

        return result

    def _stream(self, payload: dict) -> tuple[str, InferenceMetrics]:
        """Читает SSE-поток ответа, собирает текст и замеряет TTFT/TPS.

        Возвращает (полный_текст_ответа, метрики). Число токенов берём из поля
        usage/timings, которое llama.cpp присылает в конце потока; если его нет —
        приближаем по числу пришедших content-чанков (llama.cpp обычно шлёт по
        одному токену на чанк).
        """
        parts: list[str] = []
        metrics = InferenceMetrics()
        chunk_count = 0
        t_start = perf_counter()
        t_first: Optional[float] = None

        resp = self._session.post(
            self._endpoint, json=payload, stream=True, timeout=self.config.timeout
        )
        resp.raise_for_status()

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break

            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = obj.get("choices") or []
            if choices:
                piece = (choices[0].get("delta") or {}).get("content")
                if piece:
                    if t_first is None:
                        t_first = perf_counter()
                    parts.append(piece)
                    chunk_count += 1

            # Финальный чанк llama.cpp несёт usage и/или timings.
            if usage := obj.get("usage"):
                metrics.prompt_tokens = int(usage.get("prompt_tokens", 0))
                metrics.completion_tokens = int(usage.get("completion_tokens", 0))

        t_end = perf_counter()
        metrics.total_time = t_end - t_start
        metrics.ttft = (t_first - t_start) if t_first is not None else metrics.total_time
        metrics.generation_time = (t_end - t_first) if t_first is not None else 0.0
        if metrics.completion_tokens == 0:
            metrics.completion_tokens = chunk_count  # запасная оценка

        return "".join(parts), metrics

    @staticmethod
    def _parse_response(content: str) -> dict:
        """Достаёт JSON-команду из ответа модели, устойчиво к лишнему тексту."""
        result = {"command": "HOVER", "reason": "", "confidence": 0.0, "raw": content}

        text = content.strip()
        # Срезаем markdown-ограждение ```json ... ``` если оно есть.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]

        # Берём подстроку от первой { до последней } — на случай мусора вокруг.
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                result["command"] = str(parsed.get("command", "HOVER")).upper()
                result["reason"] = str(parsed.get("reason", ""))
                result["confidence"] = float(parsed.get("confidence", 0.0))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        return result

    def health_check(self) -> bool:
        """Проверяет доступность сервера. Пустой health_path => проверка пропущена."""
        if not self.config.health_path:
            return True
        try:
            r = self._session.get(
                self.config.base_url.rstrip("/") + self.config.health_path, timeout=5
            )
            return r.ok
        except requests.RequestException:
            return False
