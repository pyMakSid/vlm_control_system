from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import cv2

from src.vlm_client import VLMClient, VLMConfig
from src.prompts import MISSIONS, build_system_prompt, command_json_schema

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _natural_key(name: str) -> list:
    """Ключ натуральной сортировки: frame2 < frame10 (а не наоборот)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


class ImageFolderCapture:
    """Источник кадров из папки с изображениями.

    Повторяет интерфейс cv2.VideoCapture (isOpened/read/get/release), поэтому
    подставляется в цикл бенчмарка вместо видео без изменений логики.
    """

    def __init__(self, folder: str, fps: float = 0.0):
        self._files = sorted(
            (os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(IMAGE_EXTS)),
            key=lambda p: _natural_key(os.path.basename(p)),
        )
        self._idx = 0
        self._fps = fps

    def isOpened(self) -> bool:
        return len(self._files) > 0

    def read(self):
        while self._idx < len(self._files):
            frame = cv2.imread(self._files[self._idx])
            self._idx += 1
            if frame is not None:
                return True, frame
        return False, None

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._files))
        return 0.0

    def release(self) -> None:
        pass


def open_source(path: str, fps: float):
    """Возвращает источник кадров: папка с изображениями или видеофайл."""
    if os.path.isdir(path):
        return ImageFolderCapture(path, fps)
    return cv2.VideoCapture(path)

BACKEND_ENDPOINTS = {
    "llama": {"endpoint_path": "/v1/chat/completions", "health_path": "/health"},
    "rkllm": {"endpoint_path": "/rkllm_chat", "health_path": ""},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Бенчмарк VLM-конвейера на видеофайле")
    p.add_argument("--video", required=True,
                   help="Путь к видеофайлу (mp4) или папке с кадрами (jpg/png)")
    p.add_argument("--fps", type=float, default=0.0,
                   help="Частота кадров для папки с изображениями "
                        "(для расчёта времени кадра; 0 = не задана)")
    p.add_argument("--output", default="benchmark_result.json",
                   help="Файл JSON-отчёта")
    p.add_argument("--mission", choices=sorted(MISSIONS.keys()))
    p.add_argument("--mission-text", help="Произвольный миссионный промпт")
    p.add_argument("--backend", choices=["llama", "rkllm"], default="llama")
    p.add_argument("--url", default="http://127.0.0.1:8080",
                   help="Базовый URL сервера инференса")
    p.add_argument("--sample-every", type=int, default=1,
                   help="Обрабатывать каждый N-й кадр (1 = все кадры)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Предел числа обрабатываемых кадров (0 = без предела)")
    p.add_argument("--context-frames", type=int, default=3)
    p.add_argument("--replay-decisions", action="store_true",
                   help="Подмешивать в контекст прошлые ответы модели "
                        "(по умолчанию — только кадры, без самоподкрепления)")
    p.add_argument("--constrain-json", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Ограничивать вывод JSON-схемой команды (только llama)")
    p.add_argument("--think", action=argparse.BooleanOptionalAction, default=False,
                   help="Разрешить режим мышления reasoning-моделей (Qwen3). "
                        "По умолчанию выключено (--no-think)")
    p.add_argument("--max-image-side", type=int, default=512)
    p.add_argument("--yolo", action="store_true",
                   help="Включить предобработку YOLO")
    p.add_argument("--yolo-model", default="yolov8n.pt")
    p.add_argument("--yolo-conf", type=float, default=0.35)
    return p.parse_args()


def resolve_mission(args: argparse.Namespace) -> str:
    if args.mission_text:
        return args.mission_text
    if args.mission:
        return MISSIONS[args.mission]
    return MISSIONS["explore"]


def make_client(args: argparse.Namespace) -> VLMClient:
    ep = BACKEND_ENDPOINTS[args.backend]
    schema = None
    if args.constrain_json and args.backend == "llama":
        schema = command_json_schema()
    return VLMClient(VLMConfig(
        base_url=args.url,
        context_frames=args.context_frames,
        replay_decisions=args.replay_decisions,
        max_image_side=args.max_image_side,
        endpoint_path=ep["endpoint_path"],
        health_path=ep["health_path"],
        response_schema=schema,
        disable_thinking=not args.think,
    ))


def main() -> None:
    args = parse_args()
    mission = resolve_mission(args)
    system_prompt = build_system_prompt(args.context_frames, args.replay_decisions)

    cap = open_source(args.video, args.fps)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть '{args.video}' "
              "(пустая папка или неподдерживаемый источник)", file=sys.stderr)
        sys.exit(1)

    source_kind = "папка" if os.path.isdir(args.video) else "видео"
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    client = make_client(args)
    if not client.health_check():
        print(f"Предупреждение: сервер {args.url} недоступен.", file=sys.stderr)

    detector = None
    if args.yolo:
        from src.detector import YoloDetector
        print(f"Загрузка YOLO ({args.yolo_model})...")
        detector = YoloDetector(args.yolo_model, conf=args.yolo_conf)

    print(f"Источник ({source_kind}): {args.video} | "
          f"кадров: {video_frames} | fps: {video_fps:.2f}")
    print(f"Миссия: {mission}\nОбработка каждого {args.sample_every}-го кадра...\n")

    records: list[dict] = []
    errors = 0
    t_start = time.perf_counter()
    frame_idx = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.sample_every != 0:
            continue
        if args.max_frames and len(records) >= args.max_frames:
            break

        prompt = mission
        yolo_ms = None
        if detector is not None:
            t_y = time.perf_counter()
            detections = detector.detect(frame)
            frame = detector.annotate(frame, detections)
            prompt = f"{mission}\n{detector.summary(frame, detections)}"
            yolo_ms = (time.perf_counter() - t_y) * 1000.0

        try:
            result = client.infer(frame, system_prompt, prompt)
        except Exception as exc:
            errors += 1
            print(f"[кадр {frame_idx}] ошибка инференса: {exc}", file=sys.stderr)
            continue

        m = result["metrics"]
        rec = {
            "frame_index": frame_idx,
            "video_time_sec": round(frame_idx / video_fps, 3) if video_fps else None,
            "command": result["command"],
            "reason": result["reason"],
            "confidence": result["confidence"],
            "ttft_ms": round(m.ttft * 1000.0, 1),
            "tps": round(m.tps, 1),
            "total_time_sec": round(m.total_time, 3),
            "completion_tokens": m.completion_tokens,
            "yolo_time_ms": round(yolo_ms, 1) if yolo_ms is not None else None,
            "raw": result["raw"],
        }
        records.append(rec)
        print(f"[кадр {frame_idx:>5}] {rec['command']:<9} "
              f"conf={rec['confidence']:.2f} "
              f"TTFT={rec['ttft_ms']:.0f}ms TPS={rec['tps']:.1f} "
              f"({rec['total_time_sec']:.2f}s)")

    cap.release()
    wall = time.perf_counter() - t_start

    report = build_report(args, mission, records, errors, wall,
                          video_fps, video_frames)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    agg = report["aggregate"]
    print(f"\nОбработано кадров: {agg['processed']} | ошибок: {agg['errors']}")
    print(f"Средн. TTFT: {agg['avg_ttft_ms']:.0f} мс | "
          f"средн. TPS: {agg['avg_tps']:.1f} | "
          f"пропускная способность модели: {agg['model_fps']:.2f} кадр/с")
    print(f"Команды: {agg['command_counts']}")
    print(f"Отчёт сохранён: {args.output}")


def build_report(args, mission, records, errors, wall, video_fps, video_frames):
    """Собирает итоговый словарь отчёта с агрегатами."""
    n = len(records)
    sum_total = sum(r["total_time_sec"] for r in records)
    avg_ttft = sum(r["ttft_ms"] for r in records) / n if n else 0.0
    tps_vals = [r["tps"] for r in records if r["tps"] > 0]
    avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else 0.0
    model_fps = n / sum_total if sum_total > 0 else 0.0
    cmd_counts = Counter(r["command"] for r in records)

    return {
        "meta": {
            "video": args.video,
            "video_fps": round(video_fps, 3),
            "video_frames": video_frames,
            "backend": args.backend,
            "url": args.url,
            "mission": mission,
            "sample_every": args.sample_every,
            "context_frames": args.context_frames,
            "replay_decisions": bool(args.replay_decisions),
            "max_image_side": args.max_image_side,
            "yolo": bool(args.yolo),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "wall_time_sec": round(wall, 2),
        },
        "aggregate": {
            "processed": n,
            "errors": errors,
            "avg_ttft_ms": round(avg_ttft, 1),
            "avg_tps": round(avg_tps, 1),
            "model_fps": round(model_fps, 2),
            "total_completion_tokens": sum(r["completion_tokens"] for r in records),
            "command_counts": dict(cmd_counts),
        },
        "frames": records,
    }


if __name__ == "__main__":
    main()
