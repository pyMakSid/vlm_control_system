"""Полный конвейер одной командой со сбором артефактов в папку прогона.

Режим benchmark (по умолчанию) — полная цепочка:
  1. запуск llama-server, ожидание /health;
  2. benchmark.py  → result.json (логи ответов по кадрам);
  3. остановка сервера (дальше он не нужен);
  4. evaluate.py   → metrics.json (метрики качества);
  5. demo.py       → demo.mp4 (видео с рамками и подписями).
Всё складывается в папку прогона (--out-dir), плюс общий run.log.

Режим main — реальное время с камеры (без сбора артефактов).

Серверные флаги парсит pipeline; прочие аргументы прокидываются в benchmark.

Примеры:
    python pipeline.py --hf ggml-org/SmolVLM2-500M-Instruct-GGUF \
        --video flight.mp4 --yolo --yolo-model models/yolo11s.pt \
        --mission straight --out-dir runs/test1

    python pipeline.py --run main --source 0 --mission follow
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

from src.llama_launcher import build_command, default_threads, wait_for_health


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Полный конвейер: сервер → работа → артефакты в папку")
    # --- сервер ---
    src = p.add_mutually_exclusive_group()
    src.add_argument("--hf", default="ggml-org/SmolVLM2-500M-Instruct-GGUF")
    src.add_argument("-m", "--model", help="Путь к локальному GGUF модели")
    p.add_argument("--mmproj", help="Путь к локальному GGUF проектора")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--bin", default="llama-server")
    # --- конвейер ---
    p.add_argument("--run", choices=["benchmark", "main"], default="benchmark")
    p.add_argument("--video", help="Источник для benchmark/demo (видео или папка)")
    p.add_argument("--out-dir", help="Папка прогона (по умолчанию runs/run_<время>)")
    p.add_argument("--yolo", action="store_true", help="Включить YOLO в прогоне и демо")
    p.add_argument("--yolo-model", default="yolov8n.pt")
    p.add_argument("--yolo-conf", type=float, default=0.35)
    p.add_argument("--health-timeout", type=float, default=180.0)
    p.add_argument("--keep-server", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_known_args()


def server_command(args: argparse.Namespace) -> list[str]:
    ns = argparse.Namespace(
        bin=args.bin, model=args.model, mmproj=args.mmproj, hf=args.hf,
        host="127.0.0.1", port=args.port, ngl=args.ngl, ctx=args.ctx,
        temp=0.1, threads=default_threads(), parallel=1,
    )
    return build_command(ns)


def strip_url(extra: list[str]) -> list[str]:
    """Убирает пользовательский --url, чтобы pipeline подставил свой."""
    out, skip = [], False
    for tok in extra:
        if skip:
            skip = False
            continue
        if tok == "--url":
            skip = True
            continue
        out.append(tok)
    return out


def run_logged(cmd: list[str], log_path: str) -> int:
    """Запускает команду, дублируя вывод в консоль и в общий run.log."""
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
        return proc.returncode


def stop_server(server, keep: bool, port: int) -> None:
    if server is None or server.poll() is not None:
        return
    if keep:
        print(f"Сервер оставлен работать (PID {server.pid}, порт {port}).")
        return
    print("Остановка сервера...")
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()


def main_command(args, extra, url) -> list[str]:
    """Команда main.py: --video pipeline'а → --source, проброс YOLO-флагов."""
    cmd = [sys.executable, "main.py", "--url", url]
    if args.video:
        cmd += ["--source", args.video]   # для main источник задаётся через --source
    if args.yolo:
        cmd += ["--yolo", "--yolo-model", args.yolo_model,
                "--yolo-conf", str(args.yolo_conf)]
    return cmd + extra


def run_benchmark_chain(args, extra, url) -> int:
    """benchmark → evaluate → demo с записью артефактов в out-dir."""
    out_dir = args.out_dir or os.path.join(
        "runs", "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    result = os.path.join(out_dir, "result.json")
    metrics = os.path.join(out_dir, "metrics.json")
    video = os.path.join(out_dir, "demo.mp4")
    log = os.path.join(out_dir, "run.log")

    yolo_bench = (["--yolo", "--yolo-model", args.yolo_model,
                   "--yolo-conf", str(args.yolo_conf)] if args.yolo else [])
    yolo_demo = (["--yolo-model", args.yolo_model, "--yolo-conf", str(args.yolo_conf)]
                 if args.yolo else ["--no-yolo"])

    bench_cmd = [sys.executable, "benchmark.py", "--url", url,
                 "--video", args.video, "--output", result] + yolo_bench + extra
    eval_cmd = [sys.executable, "evaluate.py", result, "--output", metrics]
    demo_cmd = [sys.executable, "demo.py", "--video", args.video,
                "--result", result, "--output", video] + yolo_demo

    print(f"Папка прогона: {out_dir}\n[3/5] benchmark → result.json")
    rc = run_logged(bench_cmd, log)
    if rc != 0:
        print(f"benchmark завершился с кодом {rc} — цепочка прервана.",
              file=sys.stderr)
        return rc

    stop_server(_server, args.keep_server, args.port)

    print("[4/5] evaluate → metrics.json")
    run_logged(eval_cmd, log)
    print("[5/5] demo → demo.mp4")
    run_logged(demo_cmd, log)

    print("\n=== Артефакты ===")
    for name in ("result.json", "metrics.json", "demo.mp4", "run.log"):
        path = os.path.join(out_dir, name)
        mark = "✓" if os.path.exists(path) else "—"
        print(f"  {mark} {path}")
    return 0


_server = None  # глобальная ссылка для остановки из цепочки


def main() -> None:
    global _server
    args, extra = parse_args()
    extra = strip_url(extra)
    url = f"http://127.0.0.1:{args.port}"

    if args.run == "benchmark" and not args.video and not args.dry_run:
        print("Ошибка: для --run benchmark нужен --video.", file=sys.stderr)
        sys.exit(2)
    if args.run == "main" and args.video and os.path.isdir(args.video):
        print("Предупреждение: режим main не читает папку кадров (только видео/"
              "камеру). Для папки кадров используй --run benchmark.",
              file=sys.stderr)

    srv_cmd = server_command(args)
    print("=== Конвейер ===")
    print("Сервер:", " ".join(srv_cmd))
    print("Режим:", args.run)
    if args.dry_run:
        return

    print(f"\n[1/5] Запуск llama-server (порт {args.port})...")
    _server = subprocess.Popen(srv_cmd)
    rc = 0
    try:
        print("[2/5] Ожидание готовности /health...")
        if not wait_for_health(args.port, timeout=args.health_timeout):
            print("Ошибка: сервер не ответил на /health.", file=sys.stderr)
            rc = 1
            return
        print("Сервер готов.\n")

        if args.run == "benchmark":
            rc = run_benchmark_chain(args, extra, url)
        else:  # main — реальное время, без артефактов
            print("Основная работа: main\n" + "-" * 40)
            rc = subprocess.call(main_command(args, extra, url))
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        rc = 130
    finally:
        stop_server(_server, args.keep_server, args.port)
    sys.exit(rc)


if __name__ == "__main__":
    main()
