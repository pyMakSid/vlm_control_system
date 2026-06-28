from __future__ import annotations

import argparse
import sys
import time

from src.camera import FrameGrabber
from src.metrics import MetricsTracker
from src.prompts import MISSIONS, build_system_prompt, command_json_schema
from src.vlm_client import VLMClient, VLMConfig

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLM-управление БПЛА по видеопотоку")
    p.add_argument(
        "--source",
        default="0",
        help="Источник видео: индекс камеры (0), путь к файлу или RTSP/HTTP URL",
    )
    p.add_argument(
        "--mission",
        choices=sorted(MISSIONS.keys()),
        help="Готовый миссионный промпт из prompts.MISSIONS",
    )
    p.add_argument(
        "--mission-text",
        help="Произвольный миссионный промпт (приоритетнее --mission)",
    )
    p.add_argument(
        "--backend",
        choices=["llama", "rkllm"],
        default="llama",
        help="Бэкенд инференса: llama (llama.cpp) или rkllm (Rockchip NPU)",
    )
    p.add_argument(
        "--url",
        default="http://127.0.0.1:8080",
        help="Базовый URL сервера инференса",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Период между запусками инференса, с (на самом свежем кадре)",
    )
    p.add_argument(
        "--context-frames",
        type=int,
        default=3,
        help="Сколько последних кадров держать в окне контекста (1 = без истории)",
    )
    p.add_argument(
        "--replay-decisions",
        action="store_true",
        help="Подмешивать в контекст прошлые ОТВЕТЫ модели (риск "
             "самоподкрепления; по умолчанию в контекст идут только кадры)",
    )
    p.add_argument(
        "--constrain-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ограничивать вывод JSON-схемой команды (только llama; "
             "убирает мусорный формат). По умолчанию включено",
    )
    p.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Разрешить режим мышления reasoning-моделей (Qwen3 и т.п.). "
             "По умолчанию выключено (--no-think): иначе модель тратит весь "
             "бюджет токенов на скрытое рассуждение вместо ответа",
    )
    p.add_argument("--once", action="store_true", help="Обработать один кадр и выйти")
    p.add_argument(
        "--yolo",
        action="store_true",
        help="Включить предразметку кадра моделью детекции YOLO",
    )
    p.add_argument(
        "--yolo-model",
        default="yolov8n.pt",
        help="Путь/имя весов YOLO (по умолчанию лёгкая yolov8n.pt)",
    )
    p.add_argument(
        "--yolo-conf",
        type=float,
        default=0.35,
        help="Порог уверенности детекций YOLO",
    )
    return p.parse_args()


BACKEND_ENDPOINTS = {
    "llama": {"endpoint_path": "/v1/chat/completions", "health_path": "/health"},
    "rkllm": {"endpoint_path": "/rkllm_chat", "health_path": ""},
}


def make_config(args: argparse.Namespace) -> VLMConfig:
    ep = BACKEND_ENDPOINTS[args.backend]
    schema = None
    if args.constrain_json and args.backend == "llama":
        schema = command_json_schema()
    return VLMConfig(
        base_url=args.url,
        context_frames=args.context_frames,
        replay_decisions=args.replay_decisions,
        endpoint_path=ep["endpoint_path"],
        health_path=ep["health_path"],
        response_schema=schema,
        disable_thinking=not args.think,
    )


def resolve_mission(args: argparse.Namespace) -> str:
    if args.mission_text:
        return args.mission_text
    if args.mission:
        return MISSIONS[args.mission]
    return MISSIONS["explore"]


def main() -> None:
    args = parse_args()
    mission_prompt = resolve_mission(args)
    system_prompt = build_system_prompt(args.context_frames, args.replay_decisions)

    client = VLMClient(make_config(args))
    if not client.health_check():
        print(
            f"Предупреждение: сервер llama.cpp на {args.url} недоступен (/health). "
            "Проверь, что llama-server запущен.",
            file=sys.stderr,
        )

    detector = None
    if args.yolo:
        from src.detector import YoloDetector

        print(f"Загрузка YOLO ({args.yolo_model})...")
        detector = YoloDetector(args.yolo_model, conf=args.yolo_conf)

    tracker = MetricsTracker()
    print(f"Бэкенд: {args.backend} ({args.url})")
    print(f"Источник: {args.source} | Миссия: {mission_prompt}")
    print(f"Период инференса: {args.interval:.2f}s (на самом свежем кадре)")
    print(f"YOLO-предразметка: {'вкл' if detector else 'выкл'}")
    print("Нажми Ctrl+C для остановки.\n")

    try:
        grabber = FrameGrabber(args.source).start()
    except RuntimeError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        while grabber.read() is None and not grabber.ended:
            time.sleep(0.01)

        while True:
            if grabber.ended and grabber.read() is None:
                print("Кадры закончились или поток прерван.", file=sys.stderr)
                break

            frame = grabber.read()
            if frame is None:
                time.sleep(0.01)
                continue

            cycle_start = time.perf_counter()

            prompt = mission_prompt
            yolo_time = None
            if detector is not None:
                t_yolo = time.perf_counter()
                detections = detector.detect(frame)
                frame = detector.annotate(frame, detections)
                prompt = f"{mission_prompt}\n{detector.summary(frame, detections)}"
                yolo_time = time.perf_counter() - t_yolo

            try:
                result = client.infer(frame, system_prompt, prompt)
            except Exception as exc:
                print(f"Ошибка вывода VLM: {exc}", file=sys.stderr)
                if args.once:
                    break
                time.sleep(args.interval)
                continue

            m = result["metrics"]
            tracker.update(m, yolo_time=yolo_time)
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"КОМАНДА={result['command']:<9} "
                f"conf={result['confidence']:.2f} | "
                f"TTFT={m.ttft * 1000:.0f}ms TPS={m.tps:.1f} "
                f"({m.total_time:.2f}s) — {result['reason']}"
            )
            print(f"           средние: {tracker.summary_line()}")

            if args.once:
                break
            elapsed = time.perf_counter() - cycle_start
            remaining = args.interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
    finally:
        grabber.stop()
        if tracker.total:
            print(f"\nИтоговые средние по {tracker.total} кадрам:")
            print(f"  {tracker.summary_line()}")


if __name__ == "__main__":
    main()
