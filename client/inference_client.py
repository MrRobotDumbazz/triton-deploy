#!/usr/bin/env python3
"""Клиент инференса NVIDIA Triton Inference Server для triton-deploy.

Поддерживает две модели (контракт `.spec/contracts/model-io.md`, B1):
  * ``resnet50`` — классификация ImageNet-1000 (вход ``input`` [N,3,224,224],
    выходы ``probs`` [N,1000] float32 и ``preds`` [N,1] int64);
  * ``yolov5`` — детекция COCO-80 (вход ``images`` [N,3,640,640],
    выход ``output0`` [N,25200,85]).

Протокол — Triton HTTP REST (``/v2/...``) через ``tritonclient[http]``,
бинарная передача тензоров. torch и cv2 не используются: препроцессинг —
PIL+numpy, NMS — numpy.

Функции препроцессинга/постпроцессинга переиспользуются бенчмарком
(``client/benchmark.py``). Молчаливых фолбэков нет (I54): любая деградация
печатается в stderr.

CSV-лог (бонус): ``timestamp_utc,model,version,image,batch,latency_ms,ok,
prediction_summary``; путь по умолчанию — ``client/inference_log.csv``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import tritonclient.http as httpclient

HERE = Path(__file__).resolve().parent

IMAGENET_LABELS_PATH = HERE / "imagenet_classes.txt"
DEFAULT_SAMPLES_DIR = HERE / "samples"
DEFAULT_LOG_CSV = HERE / "inference_log.csv"

CSV_HEADER = [
    "timestamp_utc", "model", "version", "image",
    "batch", "latency_ms", "ok", "prediction_summary",
]

RESNET_INPUT_SIZE = 224
RESNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RESNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

YOLO_INPUT_SIZE = 640
YOLO_PAD_VALUE = 114  # pad 114/255 в нормализованных координатах
YOLO_NUM_BOXES = 25200
YOLO_NUM_CLASSES = 80

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]
assert len(COCO_CLASSES) == YOLO_NUM_CLASSES


# ---------------------------------------------------------------------------
# Загрузка изображений и препроцессинг (переиспользуются бенчмарком)
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> np.ndarray:
    """Загрузить изображение как np.uint8 HWC RGB.

    Parameters
    ----------
    path:
        Путь к файлу изображения (JPEG/PNG/...).

    Returns
    -------
    np.ndarray формы (H, W, 3), dtype uint8, каналы RGB.
    """
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    return np.asarray(rgb, dtype=np.uint8)


def preprocess_resnet50(img: np.ndarray) -> np.ndarray:
    """Препроцессинг для resnet50 (контракт B1).

    RGB HWC uint8 → resize 224×224 (bilinear) → /255 →
    normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) → CHW.

    Returns
    -------
    np.ndarray формы (3, 224, 224), dtype float32.
    """
    pil = Image.fromarray(img)
    resized = pil.resize((RESNET_INPUT_SIZE, RESNET_INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0  # HWC, [0..1]
    arr = (arr - RESNET_MEAN) / RESNET_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1))  # CHW


def preprocess_yolov5(img: np.ndarray) -> np.ndarray:
    """Препроцессинг для yolov5: letterbox(640, pad 114) → /255 → CHW.

    Returns
    -------
    np.ndarray формы (3, 640, 640), dtype float32, диапазон [0..1].
    """
    padded = letterbox(img, YOLO_INPUT_SIZE)[0]
    arr = padded.astype(np.float32) / 255.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1))  # CHW


def _letterbox_params(h: int, w: int, size: int = YOLO_INPUT_SIZE):
    """Параметры letterbox: масштаб и отступы слева/сверху (единый источник
    истины для preprocess и postprocess — координаты всегда сходятся)."""
    ratio = min(size / h, size / w)
    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    return ratio, pad_x, pad_y


def letterbox(img: np.ndarray, size: int = YOLO_INPUT_SIZE):
    """Letterbox-ресайз с сохранением пропорций и паддингом 114 (ultralytics).

    Parameters
    ----------
    img:
        Входное изображение HWC uint8 RGB.
    size:
        Целевой размер (квадрат), по умолчанию 640.

    Returns
    -------
    (padded, ratio, pad_x, pad_y):
        ``padded`` — (size, size, 3) uint8; ``ratio`` — масштаб;
        ``pad_x``/``pad_y`` — паддинг слева/сверху в пикселях.
    """
    h, w = img.shape[:2]
    ratio, pad_x, pad_y = _letterbox_params(h, w, size)
    new_w = size - 2 * pad_x
    new_h = size - 2 * pad_y
    resized = Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new("RGB", (size, size), (YOLO_PAD_VALUE,) * 3)
    padded.paste(resized, (pad_x, pad_y))
    return np.asarray(padded, dtype=np.uint8), ratio, pad_x, pad_y


# ---------------------------------------------------------------------------
# Постпроцессинг (переиспользуется бенчмарком)
# ---------------------------------------------------------------------------

def load_imagenet_labels(path: str | Path = IMAGENET_LABELS_PATH) -> list[str]:
    """Загрузить 1000 лейблов ImageNet (строка i = класс i, контракт B1)."""
    with open(path, encoding="utf-8") as fh:
        labels = [line.strip() for line in fh]
    if len(labels) != 1000:
        raise ValueError(
            f"{path}: ожидалось 1000 строк с лейблами, получено {len(labels)}"
        )
    return labels


def postprocess_resnet50(
    probs: np.ndarray,
    preds: np.ndarray,
    labels: list[str],
    k: int = 5,
) -> list[list[tuple[str, float]]]:
    """Top-k классификаций для каждого элемента батча.

    Parameters
    ----------
    probs:
        float32 [N,1000] (softmax) или [1000].
    preds:
        int64 [N,1] — argmax-индекс от модели (сверяется с probs, расхождение
        печатается в stderr — I54, без молчаливых фолбэков).
    labels:
        1000 имён классов.
    k:
        Сколько топ-классов вернуть.

    Returns
    -------
    Список по батчу; элемент — список из k пар (имя_класса, вероятность).
    """
    probs = np.atleast_2d(np.asarray(probs))
    preds = np.asarray(preds).reshape(-1)
    results: list[list[tuple[str, float]]] = []
    for i, row in enumerate(probs):
        top_idx = np.argsort(row)[::-1][:k]
        if preds.size > i and int(preds[i]) != int(top_idx[0]):
            print(
                f"WARNING: preds[{i}]={preds[i]} != argmax(probs)={top_idx[0]} "
                f"(сервер/клиент расходятся в argmax)",
                file=sys.stderr,
            )
        results.append([(labels[int(j)], float(row[j])) for j in top_idx])
    return results


def postprocess_yolov5(
    output0: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    orig_shape: tuple[int, int] | None = None,
) -> list[dict]:
    """Постпроцессинг yolov5: декод → фильтр conf → NMS → маппинг в оригинал.

    Parameters
    ----------
    output0:
        float32 [1,25200,85] или [25200,85]: строки [cx,cy,w,h,obj,cls0..79],
        сигмоиды уже применены в графе (контракт B1).
    conf:
        Порог obj·max(cls) (по контракту: conf = obj·max(cls) ≥ 0.25).
    iou:
        Порог IoU для NMS (по контракту 0.45).
    orig_shape:
        (H, W) исходного изображения — боксы возвращаются в его координатах
        (letterbox-маппинг с теми же параметрами, что в preprocess).

    Returns
    -------
    Список словарей: {"class_id", "class", "conf", "xyxy": (x1,y1,x2,y2)}.
    """
    out = np.asarray(output0, dtype=np.float32)
    if out.ndim == 3:
        out = out[0]
    boxes_cxcywh = out[:, :4]          # пиксельные координаты letterbox 640×640
    obj = out[:, 4]
    cls = out[:, 5:]
    cls_ids = np.argmax(cls, axis=1)
    scores = obj * cls.max(axis=1)     # контракт: conf = obj·max(cls)

    keep = scores >= conf
    boxes_cxcywh = boxes_cxcywh[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]
    if boxes_cxcywh.size == 0:
        return []

    xyxy = np.empty_like(boxes_cxcywh)
    xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

    keep_idx = _nms_per_class(xyxy, scores, cls_ids, iou)
    xyxy = xyxy[keep_idx]
    scores = scores[keep_idx]
    cls_ids = cls_ids[keep_idx]

    if orig_shape is not None:
        h, w = orig_shape
        ratio, pad_x, pad_y = _letterbox_params(h, w, YOLO_INPUT_SIZE)
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / ratio
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / ratio
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w - 1)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h - 1)

    return [
        {
            "class_id": int(cid),
            "class": COCO_CLASSES[int(cid)],
            "conf": float(score),
            "xyxy": tuple(round(float(v), 1) for v in box),
        }
        for cid, score, box in zip(cls_ids, scores, xyxy)
    ]


def _nms_per_class(
    boxes: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray, iou_thr: float
) -> list[int]:
    """NMS (numpy, без cv2) с раздельным подавлением по классам.

    boxes — [N,4] xyxy; возвращает индексы оставленных боксов.
    """
    keep: list[int] = []
    for cid in np.unique(cls_ids):
        idx = np.flatnonzero(cls_ids == cid)
        idx = idx[np.argsort(scores[idx])[::-1]]
        suppressed = np.zeros(idx.size, dtype=bool)
        for a in range(idx.size):
            if suppressed[a]:
                continue
            keep.append(int(idx[a]))
            box_a = boxes[idx[a]]
            area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            for b in range(a + 1, idx.size):
                if suppressed[b]:
                    continue
                box_b = boxes[idx[b]]
                ix1 = max(box_a[0], box_b[0])
                iy1 = max(box_a[1], box_b[1])
                ix2 = min(box_a[2], box_b[2])
                iy2 = min(box_a[3], box_b[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
                union = area_a + area_b - inter
                if union > 0 and inter / union > iou_thr:
                    suppressed[b] = True
    return sorted(keep)


# ---------------------------------------------------------------------------
# Транспорт Triton HTTP (бинарная передача)
# ---------------------------------------------------------------------------

def make_client(url: str) -> httpclient.InferenceServerClient:
    """Создать HTTP-клиент Triton (таймауты — чтобы отказ соединения
    не висел «вечно», а падал быстро и громко, I54)."""
    return httpclient.InferenceServerClient(
        url=url, verbose=False, connection_timeout=5.0, network_timeout=60.0
    )


def ensure_server_live(client: httpclient.InferenceServerClient, url: str) -> None:
    """Проверка здоровья: is_server_live/is_server_ready.

    При недоступности — внятная ошибка и exit 1 (стек должен подниматься
    скриптом deployment, а не клиентом).
    """
    hint = f"стек не поднят: bash deployment/run_triton.sh (проверял {url})"
    try:
        live = client.is_server_live()
        ready = client.is_server_ready()
    except Exception as exc:  # соединение не установлено вообще
        print(f"ОШИБКА: Triton недоступен — {exc}\n{hint}", file=sys.stderr)
        sys.exit(1)
    if not live or not ready:
        print(f"ОШИБКА: Triton не готов (live={live}, ready={ready})\n{hint}",
              file=sys.stderr)
        sys.exit(1)


def _served_version(result) -> str:
    """Версия, которую обслужил сервер (из метаданных ответа v2)."""
    try:
        return str(result.get_response().get("model_version", "") or "")
    except Exception:
        return ""


def infer_resnet50(
    client: httpclient.InferenceServerClient,
    batch: np.ndarray,
    model_version: str = "",
) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Запрос resnet50: input [N,3,224,224] → probs [N,1000], preds [N,1].

    Returns
    -------
    (probs, preds, latency_ms, served_version) — латентность по клиентским часам.
    """
    inp = httpclient.InferInput("input", batch.shape, "FP32")
    inp.set_data_from_numpy(batch, binary_data=True)
    outputs = [
        httpclient.InferRequestedOutput("probs", binary_data=True),
        httpclient.InferRequestedOutput("preds", binary_data=True),
    ]
    t0 = time.perf_counter()
    result = client.infer(
        "resnet50", inputs=[inp], outputs=outputs, model_version=model_version
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return (result.as_numpy("probs"), result.as_numpy("preds"), latency_ms,
            _served_version(result))


def infer_yolov5(
    client: httpclient.InferenceServerClient,
    batch: np.ndarray,
    model_version: str = "",
) -> tuple[np.ndarray, float, str]:
    """Запрос yolov5: images [N,3,640,640] → output0 [N,25200,85].

    Returns
    -------
    (output0, latency_ms, served_version).
    """
    inp = httpclient.InferInput("images", batch.shape, "FP32")
    inp.set_data_from_numpy(batch, binary_data=True)
    outputs = [httpclient.InferRequestedOutput("output0", binary_data=True)]
    t0 = time.perf_counter()
    result = client.infer(
        "yolov5", inputs=[inp], outputs=outputs, model_version=model_version
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return result.as_numpy("output0"), latency_ms, _served_version(result)


# ---------------------------------------------------------------------------
# CSV-лог (бонус)
# ---------------------------------------------------------------------------

def append_inference_csv(
    path: str | Path,
    *,
    model: str,
    version: str,
    image: str,
    batch: int,
    latency_ms: float,
    ok: bool,
    prediction_summary: str,
) -> None:
    """Append-строка в CSV-лог инференса; создаёт файл с заголовком."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    row = [
        datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        model,
        version or "",
        image,
        str(batch),
        f"{latency_ms:.2f}",
        "true" if ok else "false",
        prediction_summary,
    ]
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(CSV_HEADER)
        writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_images(paths: list[str | Path]) -> list[Path]:
    """Развести аргументы (файлы и/или каталоги) в отсортированный список
    изображений *.jpg/*.jpeg/*.png (регистронезависимо)."""
    exts = {".jpg", ".jpeg", ".png"}
    found: set[Path] = set()
    for p in map(Path, paths):
        if p.is_dir():
            found.update(f for f in p.iterdir() if f.suffix.lower() in exts)
        elif p.is_file():
            if p.suffix.lower() not in exts:
                print(f"ОШИБКА: неподдерживаемый формат {p} "
                      f"(нужен jpg/jpeg/png)", file=sys.stderr)
                sys.exit(2)
            found.add(p)
        else:
            print(f"ОШИБКА: путь не найден: {p}", file=sys.stderr)
            sys.exit(2)
    return sorted(found)


def _print_resnet50(
    name: str, version: str, latency_ms: float,
    topk: list[tuple[str, float]], batch: int = 1,
) -> None:
    print(f"{name} (v{version or 'default'}, batch={batch}, "
          f"{latency_ms:.1f} ms):")
    for rank, (label, prob) in enumerate(topk, start=1):
        print(f"  {rank}. {label:<30s} {prob * 100:.2f}%")


def _print_yolov5(
    name: str, version: str, latency_ms: float, detections: list[dict],
) -> None:
    print(f"{name} (v{version or 'default'}, {latency_ms:.1f} ms): "
          f"{len(detections)} box(es)")
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        print(f"  {det['class']:<15s} conf={det['conf']:.3f}  "
              f"[x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}]")


def _summary_resnet50(topk: list[tuple[str, float]]) -> str:
    return "; ".join(f"{label}:{prob * 100:.2f}%" for label, prob in topk[:3])


def _summary_yolov5(detections: list[dict]) -> str:
    if not detections:
        return "no detections"
    parts = [f"{d['class']} {d['conf']:.2f}" for d in detections[:5]]
    return f"{len(detections)} box(es): " + "; ".join(parts)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference_client.py",
        description="Клиент инференса Triton: resnet50 (ImageNet-1000) и "
                    "yolov5 (COCO-80). Контракт: .spec/contracts/model-io.md",
    )
    parser.add_argument(
        "--model", required=True, choices=["resnet50", "yolov5"],
        help="какую модель запросить",
    )
    parser.add_argument(
        "images", nargs="*", default=None,
        help=f"файлы и/или каталоги изображений (jpg/jpeg/png); "
             f"по умолчанию {DEFAULT_SAMPLES_DIR}",
    )
    parser.add_argument(
        "--images", dest="images_flag", nargs="+", default=None,
        help="то же, что позиционные аргументы (для совместимости с README)",
    )
    parser.add_argument(
        "--url", default=os.environ.get("TRITON_URL", "localhost:8000"),
        help="URL Triton HTTP (env TRITON_URL, по умолчанию localhost:8000)",
    )
    parser.add_argument(
        "--version", default="",
        help="версия модели → /v2/models/<m>/versions/<v>/infer "
             "(пусто = по version_policy сервера)",
    )
    parser.add_argument(
        "--top-k", dest="topk", type=int, default=5,
        help="сколько топ-классов печатать для resnet50 (по умолчанию 5)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="порог conf = obj*max(cls) для yolov5 (по умолчанию 0.25)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="порог IoU для NMS yolov5 (по умолчанию 0.45)",
    )
    parser.add_argument(
        "--log-csv", default=str(DEFAULT_LOG_CSV),
        help=f"путь CSV-лога инференса (по умолчанию {DEFAULT_LOG_CSV})",
    )
    parser.add_argument(
        "--no-log", action="store_true", help="отключить CSV-лог",
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="размер батча для resnet50 (>1 = один запрос на N картинок; "
             "для yolov5 не поддерживается)",
    )
    return parser


def run_resnet50(args, client: httpclient.InferenceServerClient,
                 images: list[Path]) -> None:
    labels = load_imagenet_labels()
    tensors = {p: preprocess_resnet50(load_image(p)) for p in images}
    batch_size = max(1, args.batch)
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]
        batch = np.stack([tensors[p] for p in chunk])
        try:
            probs, preds, latency_ms, served = infer_resnet50(
                client, batch, model_version=args.version)
            per_image = postprocess_resnet50(probs, preds, labels,
                                             k=args.topk)
        except Exception as exc:
            print(f"ОШИБКА инференса resnet50 ({chunk[0].name}.."
                  f"{chunk[-1].name}, batch={len(chunk)}): {exc}",
                  file=sys.stderr)
            if not args.no_log:
                append_inference_csv(
                    args.log_csv, model="resnet50", version=args.version,
                    image=f"{chunk[0].name}..{chunk[-1].name}",
                    batch=len(chunk), latency_ms=0.0, ok=False,
                    prediction_summary=f"error: {exc}")
            continue
        for path, topk in zip(chunk, per_image):
            _print_resnet50(path.name, served or args.version, latency_ms,
                            topk, batch=len(chunk))
            if not args.no_log:
                append_inference_csv(
                    args.log_csv, model="resnet50",
                    version=served or args.version,
                    image=path.name, batch=len(chunk),
                    latency_ms=latency_ms / len(chunk), ok=True,
                    prediction_summary=_summary_resnet50(topk))


def run_yolov5(args, client: httpclient.InferenceServerClient,
               images: list[Path]) -> None:
    for path in images:
        img = load_image(path)
        orig_shape = img.shape[:2]
        batch = np.stack([preprocess_yolov5(img)])
        try:
            output0, latency_ms, served = infer_yolov5(
                client, batch, model_version=args.version)
            detections = postprocess_yolov5(
                output0, conf=args.conf, iou=args.iou,
                orig_shape=orig_shape)
        except Exception as exc:
            print(f"ОШИБКА инференса yolov5 ({path.name}): {exc}",
                  file=sys.stderr)
            if not args.no_log:
                append_inference_csv(
                    args.log_csv, model="yolov5", version=args.version,
                    image=path.name, batch=1, latency_ms=0.0, ok=False,
                    prediction_summary=f"error: {exc}")
            continue
        _print_yolov5(path.name, served or args.version, latency_ms,
                      detections)
        if not args.no_log:
            append_inference_csv(
                args.log_csv, model="yolov5", version=served or args.version,
                image=path.name, batch=1, latency_ms=latency_ms, ok=True,
                prediction_summary=_summary_yolov5(detections))


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.batch > 1 and args.model != "resnet50":
        print("ОШИБКА: --batch > 1 поддерживается только для resnet50",
              file=sys.stderr)
        return 2

    paths = args.images_flag + args.images if args.images_flag else (
        args.images if args.images else [DEFAULT_SAMPLES_DIR])
    images = collect_images(paths)
    if not images:
        print(f"ОШИБКА: изображения не найдены ({' '.join(map(str, paths))})",
              file=sys.stderr)
        return 2

    client = make_client(args.url)
    ensure_server_live(client, args.url)

    if args.model == "resnet50":
        run_resnet50(args, client, images)
    else:
        run_yolov5(args, client, images)

    if not args.no_log:
        print(f"CSV-лог: {args.log_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
