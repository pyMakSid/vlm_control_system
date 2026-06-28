from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen


def default_threads() -> int:
    return os.cpu_count() or 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Запуск llama-server с VLM")
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--hf",
        default="ggml-org/SmolVLM2-500M-Instruct-GGUF",
        help="Репозиторий Hugging Face (модель + mmproj скачаются автоматически)",
    )
    src.add_argument("-m", "--model", help="Путь к локальному GGUF модели")

    p.add_argument("--mmproj", help="Путь к локальному GGUF проектора (с -m/--model)")
    p.add_argument("--host", default="0.0.0.0", help="Адрес сервера")
    p.add_argument("--port", type=int, default=8080, help="Порт сервера")
    p.add_argument("--ngl", type=int, default=99, help="Слоёв на GPU/Metal (0 = CPU)")
    p.add_argument("--ctx", type=int, default=4096, help="Размер контекста")
    p.add_argument("--temp", type=float, default=0.1, help="Температура сэмплинга")
    p.add_argument(
        "-t", "--threads", type=int, default=default_threads(), help="Потоки CPU"
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Число слотов (параллельных запросов). Для одного видеопотока — 1",
    )
    p.add_argument(
        "--bin", default="llama-server", help="Имя/путь бинаря llama-server"
    )
    p.add_argument(
        "--wait",
        action="store_true",
        help="Ждать готовности /health и сообщить об успешном старте",
    )
    return p.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [args.bin]

    if args.model:
        cmd += ["-m", args.model]
        if args.mmproj:
            cmd += ["--mmproj", args.mmproj]
        else:
            print(
                "Предупреждение: указан -m без --mmproj. Без проектора зрения "
                "модель не сможет обрабатывать изображения (см. SETUP.md).",
                file=sys.stderr,
            )
    else:
        cmd += ["-hf", args.hf]

    cmd += [
        "--host", args.host,
        "--port", str(args.port),
        "-ngl", str(args.ngl),
        "-c", str(args.ctx),
        "-fa", "on",
        "--temp", str(args.temp),
        "-t", str(args.threads),
        "--parallel", str(args.parallel),
    ]
    return cmd


def wait_for_health(port: int, timeout: float = 120.0) -> bool:
    """Опрашивает /health пока сервер не ответит ok или не истечёт таймаут."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def main() -> None:
    args = parse_args()

    if shutil.which(args.bin) is None and not os.path.exists(args.bin):
        print(
            f"Ошибка: '{args.bin}' не найден. Установи llama.cpp "
            "(см. SETUP.md) или укажи путь через --bin.",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = build_command(args)
    source = args.model or f"-hf {args.hf}"
    print(
        f"Запуск llama-server: {source} host={args.host} port={args.port} "
        f"ngl={args.ngl} ctx={args.ctx} threads={args.threads} "
        f"parallel={args.parallel}"
    )
    print("Команда:", " ".join(cmd), "\n")

    proc = subprocess.Popen(cmd)

    try:
        if args.wait:
            if wait_for_health(args.port):
                print(f"\nСервер готов: http://127.0.0.1:{args.port}")
            else:
                print("\nСервер не ответил на /health за отведённое время.",
                      file=sys.stderr)
        proc.wait()
    except KeyboardInterrupt:
        print("\nОстановка сервера...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    sys.exit(proc.returncode or 0)


if __name__ == "__main__":
    main()
