from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter

from src.prompts import ALLOWED_COMMANDS

COMMAND_KEYWORDS = {
    "FORWARD": ["вперёд", "вперед", "прям"],
    "BACKWARD": ["назад"],
    "LEFT": ["влево", "лев"],
    "RIGHT": ["вправо", "прав"],
    "UP": ["вверх", "выше", "подним", "набор высот"],
    "DOWN": ["вниз", "ниже", "сниж"],
    "YAW_LEFT": ["рыскан", "поворот налево", "развернись влево", "влево"],
    "YAW_RIGHT": ["рыскан", "поворот направо", "развернись вправо", "вправо"],
    "HOVER": ["зависа", "завис", "удерж", "на месте"],
    "STOP": ["стоп", "останов"],
    "TAKEOFF": ["взлёт", "взлет", "взлетай"],
    "LAND": ["посад", "приземл", "садись"],
}


def parse_raw(raw: str) -> dict:
    """Пытается разобрать сырой ответ модели как JSON.

    Возвращает {"valid": bool, "obj": dict|None} — для оценки формата на уровне
    самой модели (а не уже нормализованных полей отчёта).
    """
    text = (raw or "").strip()
    if not text:
        return {"valid": False, "obj": None}
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"valid": False, "obj": None}
    try:
        obj = json.loads(text[start:end + 1])
        return {"valid": isinstance(obj, dict), "obj": obj if isinstance(obj, dict) else None}
    except (json.JSONDecodeError, ValueError):
        return {"valid": False, "obj": None}


def rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def format_metrics(frames: list[dict]) -> dict:
    """Уровень 1 — формат и валидность вывода модели."""
    n = len(frames)
    valid = empty = conf_present = reason_present = cmd_valid = 0
    for f in frames:
        parsed = parse_raw(f.get("raw", ""))
        if parsed["valid"]:
            valid += 1
            if "confidence" in parsed["obj"]:
                conf_present += 1
        if not (f.get("raw") or "").strip():
            empty += 1
        if (f.get("reason") or "").strip():
            reason_present += 1
        if f.get("command") in ALLOWED_COMMANDS:
            cmd_valid += 1
    return {
        "valid_json_rate": rate(valid, n),
        "empty_raw_rate": rate(empty, n),
        "command_in_enum_rate": rate(cmd_valid, n),
        "confidence_present_rate": rate(conf_present, n),
        "reason_present_rate": rate(reason_present, n),
    }


def distribution_metrics(frames: list[dict]) -> dict:
    """Уровень 2 — распределение и энтропия команд (вырожденность вывода)."""
    n = len(frames)
    cmds = [f.get("command") for f in frames]
    counts = Counter(cmds)
    probs = [c / n for c in counts.values()] if n else []
    entropy = abs(sum(p * math.log2(p) for p in probs if p > 0))
    max_entropy = math.log2(len(ALLOWED_COMMANDS))
    dominant = max(counts.values()) / n if n else 0.0
    return {
        "command_counts": dict(counts),
        "unique_commands_used": len(counts),
        "command_entropy_bits": round(entropy, 3),
        "command_entropy_normalized": round(entropy / max_entropy, 3) if max_entropy else 0.0,
        "dominant_command": counts.most_common(1)[0][0] if counts else None,
        "dominant_command_share": round(dominant, 4),
    }


def stability_metrics(frames: list[dict]) -> dict:
    """Уровень 3 — временна́я стабильность (залипание vs дёрганье)."""
    cmds = [f.get("command") for f in frames]
    if len(cmds) < 2:
        return {"command_change_rate": 0.0, "num_switches": 0, "longest_run": len(cmds)}
    switches = sum(1 for a, b in zip(cmds, cmds[1:]) if a != b)
    longest = run = 1
    for a, b in zip(cmds, cmds[1:]):
        run = run + 1 if a == b else 1
        longest = max(longest, run)
    return {
        "command_change_rate": round(switches / (len(cmds) - 1), 4),
        "num_switches": switches,
        "longest_run": longest,
    }


def reason_metrics(frames: list[dict]) -> dict:
    """Уровень 4 — разнообразие и согласованность обоснований."""
    reasons = [(f.get("reason") or "").strip() for f in frames]
    nonempty = [r for r in reasons if r]
    n_ne = len(nonempty)

    distinct = len(set(nonempty))
    dup_share = 0.0
    if n_ne:
        top = Counter(nonempty).most_common(1)[0][1]
        dup_share = top / n_ne

    cmd_list_hits = sum(
        1 for r in nonempty
        if "допустимые команды" in r.lower()
        or sum(c in r for c in ALLOWED_COMMANDS) >= 5
    )

    agree = checked = 0
    for f in frames:
        r = (f.get("reason") or "").lower()
        cmd = f.get("command")
        if not r or cmd not in COMMAND_KEYWORDS:
            continue
        checked += 1
        if any(kw in r for kw in COMMAND_KEYWORDS[cmd]):
            agree += 1

    return {
        "reason_distinct": distinct,
        "reason_diversity": round(distinct / n_ne, 4) if n_ne else 0.0,
        "reason_top_duplicate_share": round(dup_share, 4),
        "reason_is_command_list_rate": rate(cmd_list_hits, n_ne),
        "command_reason_agreement": round(agree / checked, 4) if checked else None,
    }


def confidence_metrics(frames: list[dict]) -> dict:
    """Уровень 5 — статистика уверенности (калибровку без меток не считаем)."""
    vals = [float(f.get("confidence", 0.0)) for f in frames]
    if not vals:
        return {}
    zero = sum(1 for v in vals if v == 0.0)
    return {
        "confidence_mean": round(statistics.mean(vals), 3),
        "confidence_std": round(statistics.pstdev(vals), 3),
        "confidence_min": round(min(vals), 3),
        "confidence_max": round(max(vals), 3),
        "confidence_zero_rate": rate(zero, len(vals)),
        "confidence_distinct_values": len(set(vals)),
    }


def performance_metrics(frames: list[dict]) -> dict:
    """Уровень 6 — сводка производительности (по полям отчёта)."""
    ttft = [f["ttft_ms"] for f in frames if "ttft_ms" in f]
    tps = [f["tps"] for f in frames if f.get("tps", 0) > 0]
    total = [f["total_time_sec"] for f in frames if "total_time_sec" in f]
    yolo = [f["yolo_time_ms"] for f in frames if f.get("yolo_time_ms") is not None]
    toks = [f["completion_tokens"] for f in frames if "completion_tokens" in f]
    out = {
        "avg_ttft_ms": round(statistics.mean(ttft), 1) if ttft else 0.0,
        "avg_tps": round(statistics.mean(tps), 1) if tps else 0.0,
        "avg_total_time_sec": round(statistics.mean(total), 3) if total else 0.0,
        "model_fps": round(len(total) / sum(total), 2) if sum(total) else 0.0,
        "avg_yolo_ms": round(statistics.mean(yolo), 1) if yolo else None,
    }
    if toks:
        top_tok, top_cnt = Counter(toks).most_common(1)[0]
        out["completion_tokens_mean"] = round(statistics.mean(toks), 1)
        # подозрение на обрыв по лимиту: одно высокое значение доминирует
        out["token_saturation_share"] = round(top_cnt / len(toks), 4)
        out["token_saturation_value"] = top_tok
    return out


def health_flags(m: dict) -> list[str]:
    """Уровень 7 — итоговые предупреждения по совокупности метрик."""
    flags = []
    fmt, dist, stab = m["format"], m["distribution"], m["stability"]
    rsn, conf, perf = m["reason"], m["confidence"], m["performance"]

    if fmt["empty_raw_rate"] > 0.1:
        flags.append(f"Пустые ответы модели: {fmt['empty_raw_rate']:.0%} "
                     "(возможен thinking-режим или обрыв — см. --no-think)")
    if fmt["valid_json_rate"] < 0.9:
        flags.append(f"Низкая доля валидного JSON: {fmt['valid_json_rate']:.0%} "
                     "(включи --constrain-json)")
    if dist["dominant_command_share"] > 0.8:
        flags.append(f"Вырожденный вывод: команда {dist['dominant_command']} в "
                     f"{dist['dominant_command_share']:.0%} кадров")
    if dist["unique_commands_used"] <= 1:
        flags.append("Используется только одна команда — политика не реагирует на сцену")
    if stab["command_change_rate"] == 0.0 and len(m.get("_cmds", [])) != 1:
        flags.append("Залипание: команда не меняется между кадрами")
    if rsn["reason_top_duplicate_share"] > 0.5:
        flags.append(f"Дословное повторение reason: "
                     f"{rsn['reason_top_duplicate_share']:.0%} (самоподкрепление; "
                     "проверь, что replay_decisions выключен)")
    if rsn["reason_is_command_list_rate"] > 0.1:
        flags.append("reason копирует список команд вместо обоснования")
    if conf.get("confidence_std", 1) == 0.0:
        flags.append("confidence — константа (модель не оценивает уверенность)")
    if perf.get("token_saturation_share", 0) > 0.5 and \
            fmt["empty_raw_rate"] > 0.1:
        flags.append(f"Токены упёрлись в лимит ({perf.get('token_saturation_value')}) "
                     "при пустых ответах — модель «думает» вместо ответа")
    if not flags:
        flags.append("Грубых аномалий не обнаружено")
    return flags


def evaluate(report: dict) -> dict:
    frames = report.get("frames", [])
    metrics = {
        "processed": len(frames),
        "format": format_metrics(frames),
        "distribution": distribution_metrics(frames),
        "stability": stability_metrics(frames),
        "reason": reason_metrics(frames),
        "confidence": confidence_metrics(frames),
        "performance": performance_metrics(frames),
    }
    metrics["_cmds"] = [f.get("command") for f in frames]
    metrics["flags"] = health_flags(metrics)
    metrics.pop("_cmds", None)
    return metrics


def print_report(meta: dict, m: dict) -> None:
    print("=" * 64)
    print("ОЦЕНКА КАЧЕСТВА ВЫВОДА (без эталонной разметки)")
    print("=" * 64)
    print(f"Кадров: {m['processed']} | бэкенд: {meta.get('backend')} | "
          f"YOLO: {meta.get('yolo')}")
    print(f"Миссия: {meta.get('mission', '')[:70]}...")

    def section(title, d):
        print(f"\n— {title} —")
        for k, v in d.items():
            print(f"    {k}: {v}")

    section("Формат", m["format"])
    section("Распределение команд", m["distribution"])
    section("Стабильность", m["stability"])
    section("Обоснования (reason)", m["reason"])
    section("Уверенность", m["confidence"])
    section("Производительность", m["performance"])

    print("\n— Флаги —")
    for fl in m["flags"]:
        print(f"    • {fl}")
    print("=" * 64)


def main() -> None:
    p = argparse.ArgumentParser(description="Безразметочная оценка result.json")
    p.add_argument("report", help="Путь к JSON-отчёту бенчмарка (result.json)")
    p.add_argument("--output", help="Сохранить метрики в JSON-файл")
    args = p.parse_args()

    with open(args.report, encoding="utf-8") as f:
        report = json.load(f)

    metrics = evaluate(report)
    print_report(report.get("meta", {}), metrics)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nМетрики сохранены: {args.output}")


if __name__ == "__main__":
    main()
