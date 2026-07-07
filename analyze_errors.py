"""
analyze_errors.py
=================
Phan tich loi detection tren tap test that co label YOLO.

Script nay bo sung cho eval_per_class.py: thay vi chi xem metric tong, no xuat
CSV cac false negative / false positive va gan nhom loi de biet can bo sung du
lieu nao cho vong train tiep theo.

Vi du:
    python analyze_errors.py --weights runs_signature/sweep_yolo11s_imgsz1024/weights/best.pt --data real_test/data.yaml --split test --imgsz 1280 --conf 0.15 --save_crops error_crops

Hoac truyen truc tiep folder:
    python analyze_errors.py --weights best.pt --images_dir real_test/images --labels_dir real_test/labels
"""

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = {0: "signature", 1: "stamp"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_simple_data_yaml(data_path, split):
    """Doc data.yaml YOLO don gian ma khong can PyYAML."""
    data_file = Path(data_path)
    values = {}
    with data_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key in {"path", "train", "val", "test"}:
                values[key] = value

    root = Path(values.get("path", data_file.parent))
    if not root.is_absolute():
        root = (data_file.parent / root).resolve()

    if split not in values:
        raise ValueError(f"Khong thay '{split}:' trong {data_path}")

    images_dir = Path(values[split])
    if not images_dir.is_absolute():
        images_dir = (root / images_dir).resolve()
    return images_dir


def infer_labels_dir(images_dir):
    parts = list(Path(images_dir).parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts)
    return Path(images_dir).parent / "labels"


def list_images(images_dir):
    images = []
    for path in Path(images_dir).rglob("*"):
        if path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
    return sorted(images)


def load_yolo_labels(label_path, img_w, img_h):
    labels = []
    if not Path(label_path).exists():
        return labels
    with Path(label_path).open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = [float(v) for v in parts[1:]]
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            labels.append({
                "class_id": cls,
                "box": [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)],
            })
    return labels


def result_to_predictions(result, conf_threshold):
    preds = []
    boxes = result.boxes
    if boxes is None:
        return preds
    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        cls = int(box.cls[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        preds.append({"class_id": cls, "confidence": conf, "box": [x1, y1, x2, y2]})
    return preds


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def overlap_ratio(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    return inter / area_a if area_a > 0 else 0.0


def match_predictions(gt_items, pred_items, iou_threshold):
    candidates = []
    for gi, gt in enumerate(gt_items):
        for pi, pred in enumerate(pred_items):
            if gt["class_id"] != pred["class_id"]:
                continue
            iou = box_iou(gt["box"], pred["box"])
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))

    candidates.sort(reverse=True)
    matched_gt = set()
    matched_pred = set()
    for _, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)

    return matched_gt, matched_pred


def size_group(box, img_w, img_h):
    x1, y1, x2, y2 = box
    area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1, img_w * img_h)
    if area_ratio < 0.003:
        return "small", area_ratio
    if area_ratio < 0.015:
        return "medium", area_ratio
    return "large", area_ratio


def orientation_group(img_w, img_h):
    ratio = img_w / max(1, img_h)
    if ratio > 1.15:
        return "landscape"
    if ratio < 0.85:
        return "portrait"
    return "squareish"


def stamp_red_signal(image_bgr, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    crop = image_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return 0.0
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    red = rgb[..., 0]
    other = np.maximum(rgb[..., 1], rgb[..., 2])
    return float(np.maximum(0, red - other).mean() / 255.0)


def build_notes(error_type, cls_id, box, img_w, img_h, gt_items, image_bgr):
    notes = []
    group, area_ratio = size_group(box, img_w, img_h)
    if cls_id == 0 and error_type == "false_negative" and group == "small":
        notes.append("miss_signature_small")
    if cls_id == 0:
        stamp_boxes = [gt["box"] for gt in gt_items if gt["class_id"] == 1]
        if any(overlap_ratio(box, stamp_box) > 0.15 for stamp_box in stamp_boxes):
            notes.append("signature_overlapped_by_stamp")
    if cls_id == 1:
        red_signal = stamp_red_signal(image_bgr, box)
        if red_signal < 0.04:
            notes.append("stamp_low_red_or_faint")
    if error_type == "false_positive" and cls_id == 0:
        notes.append("possible_handwriting_or_line_as_signature")
    if error_type == "false_positive" and cls_id == 1:
        notes.append("possible_logo_or_line_as_stamp")
    if orientation_group(img_w, img_h) == "landscape":
        notes.append("landscape_image")
    return group, area_ratio, ";".join(notes) if notes else "general"


def save_error_crop(save_dir, image_bgr, image_path, row):
    if not save_dir:
        return ""
    x1, y1, x2, y2 = [int(round(float(v))) for v in row["box_xyxy"].split()]
    crop = image_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return ""
    out_dir = Path(save_dir) / row["error_type"] / row["class_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(image_path).stem}_{row['index']:03d}.jpg"
    cv2.imwrite(str(out_path), crop)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Duong dan best.pt")
    parser.add_argument("--data", help="data.yaml cua tap test/val")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--images_dir", help="Thu muc anh neu khong dung --data")
    parser.add_argument("--labels_dir", help="Thu muc label YOLO neu khong dung --data")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument("--out_csv", default="error_analysis.csv")
    parser.add_argument("--save_crops", default=None,
                        help="Neu truyen folder, script se luu crop cac loi de active learning.")
    args = parser.parse_args()

    if args.data:
        images_dir = read_simple_data_yaml(args.data, args.split)
        labels_dir = Path(args.labels_dir) if args.labels_dir else infer_labels_dir(images_dir)
    else:
        if not args.images_dir:
            raise ValueError("Can --data hoac --images_dir")
        images_dir = Path(args.images_dir)
        labels_dir = Path(args.labels_dir) if args.labels_dir else infer_labels_dir(images_dir)

    images = list_images(images_dir)
    if not images:
        raise ValueError(f"Khong co anh trong {images_dir}")

    model = YOLO(args.weights)
    rows = []
    summary = Counter()
    matched_count = 0
    total_gt_boxes = 0
    total_pred_boxes = 0

    for image_path in images:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        img_h, img_w = image_bgr.shape[:2]
        rel = image_path.relative_to(images_dir)
        label_path = labels_dir / rel.with_suffix(".txt")
        gt_items = load_yolo_labels(label_path, img_w, img_h)
        total_gt_boxes += len(gt_items)

        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            max_det=args.max_det,
            verbose=False,
        )
        pred_items = result_to_predictions(results[0], args.conf)
        total_pred_boxes += len(pred_items)
        matched_gt, matched_pred = match_predictions(gt_items, pred_items, args.iou)
        matched_count += len(matched_gt)

        for gi, gt in enumerate(gt_items):
            if gi in matched_gt:
                continue
            cls_id = gt["class_id"]
            group, area_ratio, notes = build_notes(
                "false_negative", cls_id, gt["box"], img_w, img_h, gt_items, image_bgr
            )
            rows.append({
                "index": len(rows),
                "image": str(image_path),
                "error_type": "false_negative",
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                "confidence": "",
                "box_xyxy": " ".join(f"{v:.1f}" for v in gt["box"]),
                "image_size": f"{img_w}x{img_h}",
                "orientation": orientation_group(img_w, img_h),
                "size_group": group,
                "area_ratio": f"{area_ratio:.6f}",
                "notes": notes,
                "crop_path": "",
            })

        for pi, pred in enumerate(pred_items):
            if pi in matched_pred:
                continue
            cls_id = pred["class_id"]
            group, area_ratio, notes = build_notes(
                "false_positive", cls_id, pred["box"], img_w, img_h, gt_items, image_bgr
            )
            rows.append({
                "index": len(rows),
                "image": str(image_path),
                "error_type": "false_positive",
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                "confidence": f"{pred['confidence']:.4f}",
                "box_xyxy": " ".join(f"{v:.1f}" for v in pred["box"]),
                "image_size": f"{img_w}x{img_h}",
                "orientation": orientation_group(img_w, img_h),
                "size_group": group,
                "area_ratio": f"{area_ratio:.6f}",
                "notes": notes,
                "crop_path": "",
            })

        if args.save_crops:
            image_rows = [row for row in rows if row["image"] == str(image_path)]
            for row in image_rows:
                row["crop_path"] = save_error_crop(args.save_crops, image_bgr, image_path, row)

    for row in rows:
        summary[(row["error_type"], row["class_name"], row["notes"])] += 1

    out_path = Path(args.out_csv)
    if out_path.parent and str(out_path.parent) != ".":
        out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index", "image", "error_type", "class_id", "class_name", "confidence",
        "box_xyxy", "image_size", "orientation", "size_group", "area_ratio",
        "notes", "crop_path",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Da quet {len(images)} anh. GT boxes: {total_gt_boxes}. "
        f"Pred boxes: {total_pred_boxes}. Matched boxes: {matched_count}. Loi: {len(rows)}"
    )
    print(f"Da luu CSV: {out_path}")
    print("\nTop nhom loi:")
    for (error_type, class_name, notes), count in summary.most_common(20):
        print(f"  {count:4d} | {error_type:14s} | {class_name:10s} | {notes}")


if __name__ == "__main__":
    main()
