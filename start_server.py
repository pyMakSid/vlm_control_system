from __future__ import annotations

import sys



def main() -> None:
    argv = sys.argv[1:]

    backend = "llama"
    if "--backend" in argv:
        i = argv.index("--backend")
        try:
            backend = argv[i + 1]
        except IndexError:
            print("Ошибка: --backend требует значение (llama|rkllm)", file=sys.stderr)
            sys.exit(2)
        del argv[i : i + 2]

    if backend not in ("llama", "rkllm"):
        print(f"Неизвестный бэкенд '{backend}'. Допустимо: llama, rkllm.",
              file=sys.stderr)
        sys.exit(2)

    sys.argv = [f"start_server.py ({backend})"] + argv

    if backend == "llama":
        from llama_launcher import main as run
    else:
        from rkllm_server import main as run
    run()


if __name__ == "__main__":
    main()
