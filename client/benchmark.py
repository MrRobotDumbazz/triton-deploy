#!/usr/bin/env python3
"""Бенчмарк Triton before/after для triton-deploy (контракт B1, раздел
«Протокол бенчмарка before/after»).

Прогоняет resnet50 (по умолчанию версии ``1,2`` — before=v1 fp32,
after=v2 int8) и/или yolov5 по списку батчей, замеряет латентность per
request по клиентским часам и throughput; результаты — только из
реальных прогонов (E8). Препроцессинг/постпроцессинг и транспорт
импортируются из ``inference_client`` — код не дублируется.

Если сервер недоступен — громкая ошибка и выход без записи файлов
(никаких выдуманных чисел, E8/П5; I54).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from inference_client import (
    DEFAULT_SAMPLES_DIR,
    collect_images,
    ensure_server_live,
    infer_resnet50,
    infer_yolov5,
    load_image,
    make_client,
    preprocess_resnet50,
    preprocess_yolov5,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUT_MD = HERE / "benchmark_results.md"
DEFAULT_OUT_JSON = HERE / "benchmark_results.json"

CONTRACT_IMAGE = "nvcr.io/nvidia/tritonserver:23.12-py3 (образ из контракта B1)"

# Контракт B1: итераций 50 (resnet50) / 20 (yolov5); max_batch_size:
# resnet50 32, yolov5 8.
DEFAULT_ITERS = {"resnet50": 50, "yolov5": 20}
DEFAULT_BATCHES = {"resnet50": [1, 2, 4, 8, 16], "yolov5": [1, 2, 4, 8]}
MAX_BATCH = {"resnet50": 32, "yolov5": 8}


def parse_int_list(raw: str) -> list[int]:
    """'1,2,4' → [1, 2, 4] (для --versions/--batches)."""
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"пустой список: {raw!r}")
    return values


def cpu_model_name() -> str:
    """model name из /proc/cpuinfo — только для шапки отчёта (хост)."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    print("WARNING: /proc/cpuinfo недоступен — CPU-строка отчёта пустая",
          file=sys.stderr)
    return "unknown"


def ncpu() -> int:
    return os.cpu_count() or 0


def server_version(client) -> str:
    """Версия Triton из server metadata; при отказе — громко, по контракту."""
    try:
        return str(client.get_server_metadata().get("version", "?"))
    except Exception as exc:
        print(f"WARNING: get_server_metadata не удался ({exc}); "
              f"версия сервера неизвестна", file=sys.stderr)
        return "unknown"


def build_batch(model: str, images: list[Path], batch: int) -> np.ndarray:
    """Батч-тензор из выбранных картинок: препроцессинг каждой один раз,
    затем цикл до нужного размера батча (def: пул картинок «замостить»)."""
    if model == "resnet50":
        tensors = [preprocess_resnet50(load_image(p)) for p in images]
    else:
        tensors = [preprocess_yolov5(load_image(p)) for p in images]
    take = [tensors[i % len(tensors)] for i in range(batch)]
    return np.stack(take)


def run_config(client, model: str, version: int, batch: int,
               batch_input: np.ndarray, warmup: int, iters: int) -> dict:
    """warmup → iters запросов; латентность per request (perf_counter).

    Returns
    -------
    dict со статусом ok и метриками avg/p50/p95 (мс) и req/s, либо
    status=error + сообщение (числа при этом отсутствуют — E8).
    """
    infer = infer_resnet50 if model == "resnet50" else infer_yolov5
    version_str = str(version)
    try:
        for _ in range(warmup):
            infer(client, batch_input, model_version=version_str)
        latencies: list[float] = []
        t_start = time.perf_counter()
        for _ in range(iters):
            out = infer(client, batch_input, model_version=version_str)
            latency_ms = out[-2]  # (…, latency_ms, served_version)
            latencies.append(latency_ms)
        total_s = time.perf_counter() - t_start
    except Exception as exc:
        return {"model": model, "version": version, "batch": batch,
                "status": "error", "error": str(exc)}

    arr = np.asarray(latencies, dtype=np.float64)
    return {
        "model": model,
        "version": version,
        "batch": batch,
        "iters": iters,
        "warmup": warmup,
        "status": "ok",
        "avg_ms": round(float(arr.mean()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "rps": round(iters / total_s, 2),
    }


def merge_results(json_path: Path, meta: dict,
                  results: list[dict]) -> list[dict]:
    """Слияние с прошлыми прогонами: результаты той же модели заменяются,
    остальных моделей — сохраняются (отчёт общий на все модели)."""
    all_results = list(results)
    if json_path.exists():
        try:
            prev = json.loads(json_path.read_text(encoding="utf-8"))
            keep = [r for r in prev.get("results", [])
                    if r.get("model") != meta["model"]]
            all_results = keep + all_results
        except Exception as exc:
            print(f"[bench] ПРЕДУПРЕЖДЕНИЕ: не прочитан прошлый {json_path} "
                  f"({exc}) — отчёт только за текущий прогон", file=sys.stderr)
    order = {"resnet50": 0, "yolov5": 1}
    return sorted(all_results,
                  key=lambda r: (order.get(r.get("model"), 9),
                                 r.get("version", 0), r.get("batch", 0)))


def write_markdown(path: Path, meta: dict, results: list[dict]) -> None:
    """Markdown-отчёт: шапка + таблица (модель, версия, batch, avg/p50/p95,
    req/s); упавшие конфигурации помечаются ERROR без чисел."""
    models = sorted({r["model"] for r in results})
    lines = [
        f"# Benchmark: triton-deploy — {', '.join(models)} (before/after)",
        "",
        f"- Дата (UTC): {meta['date_utc']}",
        f"- CPU: {meta['cpu']} ({meta['ncpu']} ядер)",
        f"- Triton: {meta['triton_image']}; server {meta['server_version']}; "
        f"{meta['url']}",
        f"- Клиент: tritonclient[http], бинарная передача; латентность — "
        f"клиентские часы per request",
        f"- Примечание: before = v1 (fp32), after = v2 (int8) для resnet50; "
        f"yolov5 — v1 fp32. warmup={meta['warmup']}, "
        f"итераций (последний прогон, {meta['model']})={meta['iters_default']}",
        "",
        "| модель | версия | batch | avg, мс | p50, мс | p95, мс | req/s |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        version = f"v{r['version']}"
        if r["model"] == "resnet50":
            version += " (before, fp32)" if r["version"] == 1 else \
                       " (after, int8)" if r["version"] == 2 else ""
        if r["status"] == "ok":
            lines.append(
                f"| {r['model']} | {version} | {r['batch']} "
                f"| {r['avg_ms']:.2f} | {r['p50_ms']:.2f} "
                f"| {r['p95_ms']:.2f} | {r['rps']:.2f} |"
            )
        else:
            lines.append(
                f"| {r['model']} | {version} | {r['batch']} "
                f"| ERROR | ERROR | ERROR | ERROR |"
            )
    lines.append("")
    errors = [r for r in results if r["status"] != "ok"]
    if errors:
        lines.append("## Ошибки прогонов (I54, громко)")
        for r in errors:
            lines.append(f"- {r['model']} v{r['version']} batch={r['batch']}: "
                         f"{r['error']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, meta: dict, results: list[dict]) -> None:
    payload = {"meta": meta, "results": results}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark.py",
        description="Бенчмарк Triton before/after (resnet50 v1 fp32 vs "
                    "v2 int8; yolov5). Метрики: avg/p50/p95 латентности "
                    "(мс, клиентские часы) и throughput (req/s).",
    )
    parser.add_argument(
        "--model", required=True, choices=["resnet50", "yolov5"],
        help="какую модель гонять",
    )
    parser.add_argument(
        "--versions", type=parse_int_list, default=None,
        help='список версий через запятую; по умолчанию "1,2" для resnet50 '
             "(before/after) и \"1\" для yolov5",
    )
    parser.add_argument(
        "--batches", type=parse_int_list, default=None,
        help='список размеров батча через запятую; по умолчанию '
             '"1,2,4,8,16" для resnet50 и "1,2,4,8" для yolov5 '
             "(max_batch_size 32/8 — контракт B1)",
    )
    parser.add_argument(
        "--iters", type=int, default=None,
        help="итераций на (версия, batch); по умолчанию 50 для resnet50 "
             "и 20 для yolov5 (контракт B1)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="warmup-запросов (по умолчанию 5)"
    )
    parser.add_argument(
        "--url", default=os.environ.get("TRITON_URL", "localhost:8000"),
        help="URL Triton HTTP (env TRITON_URL, по умолчанию localhost:8000)",
    )
    parser.add_argument(
        "--images", nargs="*", default=None,
        help="картинки (файлы/каталоги), из которых собирается батч "
             "(циклически); по умолчанию первая из client/samples",
    )
    parser.add_argument(
        "--out-md", default=str(DEFAULT_OUT_MD),
        help=f"markdown-отчёт (по умолчанию {DEFAULT_OUT_MD})",
    )
    parser.add_argument(
        "--out-json", default=str(DEFAULT_OUT_JSON),
        help=f"JSON-отчёт (по умолчанию {DEFAULT_OUT_JSON})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    versions = args.versions or ([1, 2] if args.model == "resnet50" else [1])
    batches = args.batches or DEFAULT_BATCHES[args.model]
    iters = args.iters or DEFAULT_ITERS[args.model]

    over_max = [b for b in batches if b > MAX_BATCH[args.model]]
    if over_max:
        print(f"ОШИБКА: batch {over_max} превышает max_batch_size="
              f"{MAX_BATCH[args.model]} для {args.model} (контракт B1)",
              file=sys.stderr)
        return 2

    paths = args.images if args.images else [sorted(DEFAULT_SAMPLES_DIR
                                                    .glob("*.jpg"))[0]]
    images = collect_images(paths)
    if not images:
        print("ОШИБКА: изображения не найдены", file=sys.stderr)
        return 2

    client = make_client(args.url)
    ensure_server_live(client, args.url)  # недоступен → exit 1, ничего не пишем

    results: list[dict] = []
    for version in versions:
        for batch in batches:
            batch_input = build_batch(args.model, images, batch)
            print(f"[bench] {args.model} v{version} batch={batch}: "
                  f"warmup={args.warmup} iters={iters}...", file=sys.stderr)
            res = run_config(client, args.model, version, batch,
                             batch_input, args.warmup, iters)
            if res["status"] == "ok":
                print(f"[bench]   ok: avg={res['avg_ms']} ms, "
                      f"p50={res['p50_ms']} ms, p95={res['p95_ms']} ms, "
                      f"{res['rps']} req/s", file=sys.stderr)
            else:
                print(f"[bench]   ERROR: {res['error']}", file=sys.stderr)
            results.append(res)

    meta = {
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "cpu": cpu_model_name(),
        "ncpu": ncpu(),
        "triton_image": CONTRACT_IMAGE,
        "server_version": server_version(client),
        "url": args.url,
        "warmup": args.warmup,
        "iters_default": iters,
        "images": [p.name for p in images],
    }
    merged = merge_results(Path(args.out_json), meta, results)
    write_markdown(Path(args.out_md), meta, merged)
    write_json(Path(args.out_json), meta, merged)
    print(f"Отчёты: {args.out_md}, {args.out_json}", file=sys.stderr)
    return 0 if all(r["status"] == "ok" for r in results) else 3


if __name__ == "__main__":
    sys.exit(main())
