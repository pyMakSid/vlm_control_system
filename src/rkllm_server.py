"""RKLLM-сервер для VLM-управления БПЛА на Rockchip NPU (RK3588 и др.).

Поднимает Flask-эндпоинт /rkllm_chat, совместимый с нашим VLMClient
(--backend rkllm): принимает OpenAI-подобные messages с кадром (base64),
прогоняет кадр через RKNN vision-энкодер, отдаёт результат в RKLLM (NPU)
и стримит ответ в SSE-формате OpenAI.

Архитектура запроса:
    кадр (base64)  ->  RKNN vision-encoder  ->  image_embed (float32)
    text + system  ->  prompt
    (prompt + image_embed)  ->  rkllm_run (мультимодальный вход, NPU)
    токены ответа  ->  callback  ->  очередь  ->  SSE-стрим клиенту

============================ ВАЖНО: СВЕРИТЬ С SDK ============================
Этот файл написан по образцу airockchip/rknn-llm (flask_server.py + примеры
multimodal). РАЗМЕТКА СТРУКТУР ctypes и поля RKLLMParam/RKLLMInput МЕНЯЛИСЬ
между версиями (1.0.x / 1.1.x / 1.2.x). Перед запуском сверь определения ниже
с твоим include/rkllm.h, а препроцессинг и форму выхода энкодера — с твоей VLM.
=============================================================================
"""

from __future__ import annotations

import base64
import ctypes
import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np
from flask import Flask, Response, request

# ----------------------------------------------------------------------------
# 1. ctypes-обёртка над librkllmrt.so
#    (!) Структуры — под одну из версий SDK; сверить с rkllm.h.
# ----------------------------------------------------------------------------

RKLLM_LIB_PATH = "librkllmrt.so"  # лежит в rkllm-runtime/runtime/.../lib

# Типы входа RKLLM.
RKLLM_INPUT_PROMPT = 0
RKLLM_INPUT_TOKEN = 1
RKLLM_INPUT_EMBED = 2
RKLLM_INPUT_MULTIMODAL = 3

# Состояния колбэка.
RKLLM_RUN_NORMAL = 0
RKLLM_RUN_WAITING = 1
RKLLM_RUN_FINISH = 2
RKLLM_RUN_ERROR = 3

# Режимы инференса.
RKLLM_INFER_GENERATE = 0


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMPromptInput(ctypes.Structure):
    _fields_ = [("prompt", ctypes.c_char_p)]


class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [("embed", ctypes.POINTER(ctypes.c_float)),
                ("n_tokens", ctypes.c_size_t)]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [("input_ids", ctypes.POINTER(ctypes.c_int32)),
                ("n_tokens", ctypes.c_size_t)]


class RKLLMMultiModelInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
    ]


class _RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", RKLLMPromptInput),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModelInput),
    ]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int),
        ("input_data", _RKLLMInputUnion),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
        ("keep_history", ctypes.c_int),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int32),
        # (!) в новых версиях здесь ещё hidden-layer/logits/perf — добавь при
        # необходимости, но для генерации текста достаточно text/token_id.
    ]


# Тип колбэка: void cb(RKLLMResult*, void* userdata, int state)
CALLBACK_TYPE = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int
)


@dataclass
class RKLLMConfig:
    model_path: str
    max_context_len: int = 4096
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_k: int = 1
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    img_start: str = "<image>"      # (!) маркеры под твою VLM/чат-шаблон
    img_end: str = "</image>"
    img_content: str = "<unk>"


class RKLLM:
    """Обёртка жизненного цикла RKLLM: init -> run(stream) -> destroy."""

    def __init__(self, cfg: RKLLMConfig):
        self.cfg = cfg
        self._lib = ctypes.CDLL(RKLLM_LIB_PATH)

        self._lib.rkllm_createDefaultParam.restype = RKLLMParam
        self._lib.rkllm_init.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(RKLLMParam), CALLBACK_TYPE
        ]
        self._lib.rkllm_init.restype = ctypes.c_int
        self._lib.rkllm_run.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p,
        ]
        self._lib.rkllm_run.restype = ctypes.c_int
        self._lib.rkllm_destroy.argtypes = [ctypes.c_void_p]

        # Очередь токенов текущего запроса + сериализация (один NPU-хендл).
        self._tokens: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        # Держим ссылку на колбэк, иначе GC его удалит.
        self._cb = CALLBACK_TYPE(self._on_token)

        param = self._lib.rkllm_createDefaultParam()
        param.model_path = cfg.model_path.encode("utf-8")
        param.max_context_len = cfg.max_context_len
        param.max_new_tokens = cfg.max_new_tokens
        param.temperature = cfg.temperature
        param.top_k = cfg.top_k
        param.top_p = cfg.top_p
        param.repeat_penalty = cfg.repeat_penalty
        param.skip_special_token = True
        param.img_start = cfg.img_start.encode("utf-8")
        param.img_end = cfg.img_end.encode("utf-8")
        param.img_content = cfg.img_content.encode("utf-8")

        self._handle = ctypes.c_void_p()
        ret = self._lib.rkllm_init(
            ctypes.byref(self._handle), ctypes.byref(param), self._cb
        )
        if ret != 0:
            raise RuntimeError(f"rkllm_init вернул {ret}")

    def _on_token(self, result_ptr, userdata, state) -> None:
        """Колбэк рантайма: складывает кусочки текста в очередь."""
        if state == RKLLM_RUN_NORMAL:
            res = result_ptr.contents
            if res.text:
                self._tokens.put(res.text.decode("utf-8", errors="ignore"))
        elif state == RKLLM_RUN_FINISH:
            self._tokens.put(None)            # сигнал завершения
        elif state == RKLLM_RUN_ERROR:
            self._tokens.put(None)

    def run_stream(
        self, prompt: str,
        image_embed: Optional[np.ndarray] = None,
        n_image_tokens: int = 0,
        image_size: tuple[int, int] = (0, 0),
    ) -> Iterator[str]:
        """Запускает генерацию и стримит куски текста по мере прихода."""
        with self._lock:
            # Очистим очередь от возможных хвостов.
            while not self._tokens.empty():
                self._tokens.get_nowait()

            rk_input = RKLLMInput()
            rk_input.role = b"user"
            rk_input.enable_thinking = False

            if image_embed is not None and n_image_tokens > 0:
                rk_input.input_type = RKLLM_INPUT_MULTIMODAL
                flat = np.ascontiguousarray(image_embed.ravel(), dtype=np.float32)
                mm = RKLLMMultiModelInput()
                mm.prompt = prompt.encode("utf-8")
                mm.image_embed = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                mm.n_image_tokens = n_image_tokens
                mm.n_image = 1
                mm.image_width, mm.image_height = image_size
                rk_input.input_data.multimodal_input = mm
                # Держим буфер живым на время инференса.
                self._embed_buf = flat
            else:
                rk_input.input_type = RKLLM_INPUT_PROMPT
                rk_input.input_data.prompt_input = RKLLMPromptInput(
                    prompt=prompt.encode("utf-8")
                )

            infer = RKLLMInferParam()
            infer.mode = RKLLM_INFER_GENERATE
            infer.keep_history = 0            # без серверной истории — она у клиента

            # rkllm_run блокирует до конца генерации; колбэк наполняет очередь
            # из этого же потока, поэтому крутим run в фоне.
            done = threading.Event()

            def _worker():
                self._lib.rkllm_run(
                    self._handle, ctypes.byref(rk_input), ctypes.byref(infer), None
                )
                done.set()

            threading.Thread(target=_worker, daemon=True).start()

            while True:
                piece = self._tokens.get()
                if piece is None:
                    break
                yield piece
            done.wait(timeout=1.0)

    def destroy(self) -> None:
        self._lib.rkllm_destroy(self._handle)


# ----------------------------------------------------------------------------
# 2. RKNN vision-энкодер: кадр -> image_embed
#    (!) Препроцессинг и форма выхода — под конкретную VLM. Здесь скелет.
# ----------------------------------------------------------------------------

class VisionEncoder:
    """Кодирует кадр в эмбеддинги через RKNN-модель vision-энкодера."""

    def __init__(self, rknn_model_path: str, input_size: int = 392,
                 n_image_tokens: int = 196):
        from rknnlite.api import RKNNLite  # из rknn-toolkit-lite2 на плате

        self._input_size = input_size
        self._n_image_tokens = n_image_tokens
        self._rknn = RKNNLite()
        if self._rknn.load_rknn(rknn_model_path) != 0:
            raise RuntimeError("Не удалось загрузить RKNN vision-энкодер")
        if self._rknn.init_runtime() != 0:
            raise RuntimeError("Не удалось инициализировать RKNN runtime")

    @property
    def n_image_tokens(self) -> int:
        return self._n_image_tokens

    def encode(self, frame: np.ndarray) -> np.ndarray:
        """frame (BGR) -> image_embed float32 [n_image_tokens, hidden_dim].

        (!) Подгони предобработку под свою VLM: размер, нормировку, RGB/BGR,
        и reshape выхода RKNN под формат, который ждёт rkllm (image_embed).
        """
        size = self._input_size
        img = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Типичная нормировка CLIP/SigLIP — свериться с препроцессором модели.
        img = img.astype(np.float32) / 255.0
        inp = np.expand_dims(img, 0)  # NHWC; для некоторых моделей нужен NCHW
        outputs = self._rknn.inference(inputs=[inp])
        embed = np.asarray(outputs[0], dtype=np.float32)
        return np.ascontiguousarray(embed)


# ----------------------------------------------------------------------------
# 3. Flask-сервер, совместимый с VLMClient (--backend rkllm)
# ----------------------------------------------------------------------------

app = Flask(__name__)
_model: Optional[RKLLM] = None
_encoder: Optional[VisionEncoder] = None


def _parse_messages(messages: list[dict]) -> tuple[str, Optional[np.ndarray]]:
    """Достаёт текстовый промпт и (опционально) кадр из OpenAI-messages."""
    text_parts: list[str] = []
    frame: Optional[np.ndarray] = None

    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            text_parts.append(f"{msg.get('role', 'user')}: {content}")
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part["text"])
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:"):
                        b64 = url.split(",", 1)[1]
                        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
                        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return "\n".join(text_parts), frame


@app.get("/health")
def health() -> Response:
    return Response(json.dumps({"status": "ok"}), mimetype="application/json")


@app.post("/rkllm_chat")
def rkllm_chat() -> Response:
    body = request.get_json(force=True)
    messages = body.get("messages", [])
    prompt, frame = _parse_messages(messages)

    image_embed, n_tokens, size = None, 0, (0, 0)
    if frame is not None and _encoder is not None:
        image_embed = _encoder.encode(frame)
        n_tokens = _encoder.n_image_tokens
        size = (frame.shape[1], frame.shape[0])
        # Вставляем маркер изображения в промпт (под чат-шаблон твоей VLM).
        prompt = f"{_model.cfg.img_start}{_model.cfg.img_end}\n{prompt}"

    def generate() -> Iterator[str]:
        created = int(time.time())
        for piece in _model.run_stream(prompt, image_embed, n_tokens, size):
            chunk = {
                "id": "rkllm",
                "object": "chat.completion.chunk",
                "created": created,
                "choices": [{"index": 0, "delta": {"content": piece}}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")


def main() -> None:
    import argparse

    global _model, _encoder

    p = argparse.ArgumentParser(description="RKLLM-сервер VLM для БПЛА (NPU)")
    p.add_argument("--model", required=True, help="Путь к .rkllm модели LLM")
    p.add_argument("--vision", help="Путь к .rknn модели vision-энкодера")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temp", type=float, default=0.1)
    args = p.parse_args()

    print(f"Загрузка RKLLM: {args.model}")
    _model = RKLLM(RKLLMConfig(
        model_path=args.model,
        max_context_len=args.ctx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temp,
    ))
    if args.vision:
        print(f"Загрузка vision-энкодера RKNN: {args.vision}")
        _encoder = VisionEncoder(args.vision)
    else:
        print("ВНИМАНИЕ: vision-энкодер не задан — кадры обрабатываться не будут.")

    print(f"RKLLM-сервер слушает http://{args.host}:{args.port}")
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        _model.destroy()


if __name__ == "__main__":
    main()
