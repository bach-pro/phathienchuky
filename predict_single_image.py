"""
predict_single_image.py
=========================
Chạy model đã train trên 1 ảnh cụ thể, vẽ bounding box + tên lớp + confidence,
luu anh ket qua dung kich thuoc anh goc, va co the cat crop tu anh goc.

CÁCH DÙNG:
    uv run predict_single_image.py --weights runs/detect/runs_signature/yolo26_sig_stamp/weights/best.pt --image test/image-copy-2.png --conf 0.15 --imgsz 1024

Ảnh scan nguyên trang, chữ ký nhỏ:
    uv run predict_single_image.py --weights runs/detect/runs_signature/yolo26_sig_stamp/weights/best.pt --image test/page.jpg --conf 0.15 --imgsz 1280 --roi bottom

Ảnh rất lớn:
    uv run predict_single_image.py --weights runs/detect/runs_signature/yolo26_sig_stamp/weights/best.pt --image test/page.jpg --conf 0.15 --imgsz 1280 --tile_size 1280 --tile_overlap 0.25
Ảnh nhỏ 400x600,chữ ký lớn:
    uv run predict_single_image.py --weights runs/detect/runs_signature/yolo26_sig_stamp/weights/best.pt --image test/page.jpg --conf 0.15 --imgsz 1024
ảnh scan 1 trang A4, 307

Tham số hữu ích:
    --conf 0.15   Giảm ngưỡng confidence nếu muốn ưu tiên recall (đỡ bỏ sót chữ ký),
                  đánh đổi bằng việc có thể xuất hiện thêm box sai (false positive).
    --imgsz 1024  Phải khớp (hoặc gần khớp) với imgsz lúc train để độ chính xác tốt nhất.
    --crop_dir prediction_crops
                  Cat tung box truc tiep tu anh goc de dung cho hau xu ly.
    --roi bottom  Chi predict vung cuoi trang khi template biet chu ky/moc nam o do.
    --tile_size 1280
                  Chia anh lon thanh tile, sau do NMS de gop box trung.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

CLASS_COLORS = {0: (0, 102, 255), 1: (255, 40, 40)}   # signature=xanh, stamp=do (RGB)
CLASS_NAMES = {0: "signature", 1: "stamp"}


def clamp_box(x1, y1, x2, y2, img_w, img_h, pad=0):
    """Kep box trong bien anh goc, co the them pad pixel khi crop."""
    x1 = max(0, int(round(x1)) - pad)
    y1 = max(0, int(round(y1)) - pad)
    x2 = min(img_w, int(round(x2)) + pad)
    y2 = min(img_h, int(round(y2)) + pad)
    return x1, y1, x2, y2


def collect_predictions(result, conf_threshold, img_w, img_h):
    """
    Lay box tu Ultralytics theo toa do anh goc.

    Luu y: result.boxes.xyxy cua Ultralytics da duoc scale nguoc ve kich thuoc
    anh source ban dau, du model co letterbox/resize noi bo khi predict.
    """
    boxes = result.boxes
    detections = []

    if boxes is None or len(boxes) == 0:
        return detections

    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, img_w, img_h)
        if x2 <= x1 or y2 <= y1:
            continue
        name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        detections.append({
            "class_id": cls_id,
            "class_name": name,
            "confidence": conf,
            "box_xyxy": [x1, y1, x2, y2],
        })

    return detections


def parse_roi_xyxy(raw, img_w, img_h):
    """Parse ROI dang x1,y1,x2,y2. Neu gia tri <=1 thi coi la normalized."""
    try:
        values = [float(v.strip()) for v in raw.split(",")]
    except ValueError as exc:
        raise ValueError("--roi_xyxy phai co dang x1,y1,x2,y2") from exc
    if len(values) != 4:
        raise ValueError("--roi_xyxy phai co dung 4 gia tri")

    x1, y1, x2, y2 = values
    if max(abs(v) for v in values) <= 1.0:
        x1, x2 = x1 * img_w, x2 * img_w
        y1, y2 = y1 * img_h, y2 * img_h

    return clamp_box(x1, y1, x2, y2, img_w, img_h)


def get_roi_box(roi_mode, roi_xyxy, img_w, img_h):
    """Lay vung predict tren anh goc."""
    if roi_xyxy:
        return parse_roi_xyxy(roi_xyxy, img_w, img_h)
    if roi_mode == "bottom":
        return 0, int(img_h * 0.50), img_w, img_h
    if roi_mode == "lower_third":
        return 0, int(img_h * 0.66), img_w, img_h
    return 0, 0, img_w, img_h


def shift_detections(detections, dx, dy, img_w, img_h):
    """Doi toa do detection tu crop/tile ve anh goc."""
    shifted = []
    for det in detections:
        x1, y1, x2, y2 = det["box_xyxy"]
        x1, y1, x2, y2 = clamp_box(x1 + dx, y1 + dy, x2 + dx, y2 + dy, img_w, img_h)
        if x2 <= x1 or y2 <= y1:
            continue
        new_det = dict(det)
        new_det["box_xyxy"] = [x1, y1, x2, y2]
        shifted.append(new_det)
    return shifted


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_detections(detections, iou_threshold):
    """Loai box trung nhau khi predict bang tiling/ROI."""
    by_class = {}
    for det in detections:
        by_class.setdefault(det["class_id"], []).append(det)

    kept = []
    for class_detections in by_class.values():
        candidates = sorted(class_detections, key=lambda d: d["confidence"], reverse=True)
        while candidates:
            best = candidates.pop(0)
            kept.append(best)
            candidates = [
                det for det in candidates
                if box_iou(best["box_xyxy"], det["box_xyxy"]) < iou_threshold
            ]
    return sorted(kept, key=lambda d: d["confidence"], reverse=True)


def _axis_tile_starts(length, tile_size, step):
    if length <= tile_size:
        return [0]
    starts = []
    pos = 0
    while pos + tile_size < length:
        starts.append(pos)
        pos += step
    last = length - tile_size
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def iter_tile_boxes(width, height, tile_size, overlap):
    if tile_size <= 0 or max(width, height) <= tile_size:
        yield 0, 0, width, height
        return

    overlap = max(0.0, min(0.8, overlap))
    step = max(1, int(tile_size * (1.0 - overlap)))
    for y in _axis_tile_starts(height, tile_size, step):
        for x in _axis_tile_starts(width, tile_size, step):
            yield x, y, min(width, x + tile_size), min(height, y + tile_size)


def predict_with_roi_and_tiles(model, image_bgr, args):
    """Predict tren anh goc, co ho tro ROI va tiling, tra ve toa do anh goc."""
    img_h, img_w = image_bgr.shape[:2]
    rx1, ry1, rx2, ry2 = get_roi_box(args.roi, args.roi_xyxy, img_w, img_h)
    roi_bgr = image_bgr[ry1:ry2, rx1:rx2]
    roi_h, roi_w = roi_bgr.shape[:2]
    detections = []

    for tx1, ty1, tx2, ty2 in iter_tile_boxes(roi_w, roi_h, args.tile_size, args.tile_overlap):
        tile = roi_bgr[ty1:ty2, tx1:tx2]
        if tile.size == 0:
            continue
        results = model.predict(
            source=tile,
            conf=args.conf,
            imgsz=args.imgsz,
            max_det=args.max_det,
            verbose=False,
        )
        tile_detections = collect_predictions(results[0], args.conf, tx2 - tx1, ty2 - ty1)
        detections.extend(shift_detections(tile_detections, rx1 + tx1, ry1 + ty1, img_w, img_h))

    return nms_detections(detections, args.nms_iou), (rx1, ry1, rx2, ry2)


def draw_predictions(image_bgr, detections):
    """Ve box + label len ban sao cua anh goc, giu nguyen kich thuoc pixel."""
    img = image_bgr.copy()

    for det in detections:
        cls_id = det["class_id"]
        conf = det["confidence"]
        name = det["class_name"]
        x1, y1, x2, y2 = det["box_xyxy"]

        color_rgb = CLASS_COLORS.get(cls_id, (0, 255, 0))
        color_bgr = color_rgb[::-1]  # OpenCV dùng BGR

        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 3)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), color_bgr, -1)
        cv2.putText(img, label, (x1 + 3, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img


def save_crops_from_original(image_bgr, detections, crop_dir, source_stem, crop_pad=0):
    """Cat crop tung detection tu anh goc bang toa do goc."""
    if not crop_dir:
        return []

    out_dir = Path(crop_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_h, img_w = image_bgr.shape[:2]
    saved = []

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = clamp_box(*det["box_xyxy"], img_w, img_h, pad=crop_pad) # type: ignore
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        cls_name = det["class_name"]
        conf = det["confidence"]
        crop_name = f"{source_stem}_{idx:03d}_{cls_name}_{conf:.2f}_{x1}_{y1}_{x2}_{y2}.jpg"
        crop_path = out_dir / crop_name
        cv2.imwrite(str(crop_path), crop)
        saved.append(str(crop_path))

    return saved


def save_predictions_json(json_path, detections, image_path, img_w, img_h, crop_paths, metadata=None):
    """Luu toa do box anh goc de hau xu ly/crop lai bat cu luc nao."""
    if not json_path:
        return

    payload = {
        "image": image_path,
        "image_width": img_w,
        "image_height": img_h,
        "coordinate_space": "original_image_pixels_xyxy",
        "inference": metadata or {},
        "detections": [],
    }
    for idx, det in enumerate(detections):
        item = dict(det)
        item["crop_path"] = crop_paths[idx] if idx < len(crop_paths) else None
        payload["detections"].append(item)

    out_path = Path(json_path)
    if out_path.parent and str(out_path.parent) != ".":
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Duong dan file best.pt")
    parser.add_argument("--image", type=str, required=True, help="Duong dan anh can kiem tra")
    parser.add_argument("--conf", type=float, default=0.25, help="Nguong confidence")
    parser.add_argument("--imgsz", type=int, default=1024, help="Kich thuoc input, nen khop luc train")
    parser.add_argument("--max_det", type=int, default=300,
                          help="So box toi da moi lan predict. Giu cao vi mot don co the nhieu chu ky.")
    parser.add_argument("--roi", choices=["full", "bottom", "lower_third"], default="full",
                          help="Vung predict tren anh goc. Dung bottom/lower_third khi biet chu ky nam cuoi trang.")
    parser.add_argument("--roi_xyxy", type=str, default=None,
                          help="ROI tuy bien x1,y1,x2,y2 theo pixel hoac normalized 0..1. Uu tien hon --roi.")
    parser.add_argument("--tile_size", type=int, default=0,
                          help="Neu >0, chia ROI thanh cac tile kich thuoc nay de bat object nho tren anh lon.")
    parser.add_argument("--tile_overlap", type=float, default=0.20,
                          help="Ti le overlap giua cac tile khi --tile_size > 0.")
    parser.add_argument("--nms_iou", type=float, default=0.50,
                          help="Nguong NMS de gop box trung khi dung tiling.")
    parser.add_argument("--save", type=str, default="prediction_result.jpg",
                          help="Duong dan luu anh ket qua")
    parser.add_argument("--crop_dir", type=str, default="prediction_crops",
                          help="Thu muc luu crop cat truc tiep tu anh goc. De rong '' neu khong can crop.")
    parser.add_argument("--crop_pad", type=int, default=0,
                          help="So pixel mo rong moi box khi cat crop tu anh goc.")
    parser.add_argument("--save_json", type=str, default="prediction_boxes.json",
                          help="File JSON luu toa do box theo pixel anh goc. De rong '' neu khong can.")
    parser.add_argument("--show", action="store_true",
                          help="Hien thi anh bang matplotlib sau khi predict.")
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {args.image}")
    img_h, img_w = image_bgr.shape[:2]

    model = YOLO(args.weights)
    detections, roi_box = predict_with_roi_and_tiles(model, image_bgr, args)
    annotated = draw_predictions(image_bgr, detections)
    n_drawn = len(detections)

    save_parent = os.path.dirname(args.save)
    if save_parent:
        os.makedirs(save_parent, exist_ok=True)
    cv2.imwrite(args.save, annotated)

    source_stem = Path(args.image).stem
    crop_paths = save_crops_from_original(
        image_bgr=image_bgr,
        detections=detections,
        crop_dir=args.crop_dir,
        source_stem=source_stem,
        crop_pad=args.crop_pad,
    )
    save_predictions_json(
        args.save_json,
        detections,
        args.image,
        img_w,
        img_h,
        crop_paths,
        metadata={
            "imgsz": args.imgsz,
            "conf": args.conf,
            "max_det": args.max_det,
            "roi_box_xyxy": list(roi_box),
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "nms_iou": args.nms_iou,
        },
    )

    # In ra danh sách phát hiện được (kèm toạ độ) để dễ debug/kiểm tra
    print(f"\nAnh goc: {img_w}x{img_h}")
    print(f"ROI predict: ({roi_box[0]},{roi_box[1]})-({roi_box[2]},{roi_box[3]})")
    if args.tile_size > 0:
        print(f"Tiling: tile_size={args.tile_size}, overlap={args.tile_overlap:.2f}, nms_iou={args.nms_iou:.2f}")
    print(f"\nPhat hien {n_drawn} doi tuong (conf >= {args.conf}):")
    for det in detections:
        x1, y1, x2, y2 = det["box_xyxy"]
        print(f"  - {det['class_name']:10s} conf={det['confidence']:.3f}  box=({x1},{y1})-({x2},{y2})")

    print(f"\nDa luu anh ket qua dung kich thuoc goc: {args.save}")
    if args.crop_dir:
        print(f"Da luu {len(crop_paths)} crop tu anh goc vao: {args.crop_dir}")
    if args.save_json:
        print(f"Da luu toa do box anh goc vao: {args.save_json}")

    if args.show:
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 14))
        plt.imshow(annotated_rgb)
        plt.axis("off")
        plt.title(f"{args.image} - {n_drawn} phat hien (conf >= {args.conf})")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
