"""
predict_single_image.py
=========================
Chạy model đã train trên 1 ảnh cụ thể, vẽ bounding box + tên lớp + confidence,
hiển thị bằng matplotlib và lưu ảnh kết quả ra file.

CÁCH DÙNG:
    python predict_single_image.py --weights runs\\detect\\runs_signature\\yolov8_sig_stamp-4\\weights\\best.pt --image test/image-copy-2.png --conf 0.15 --imgsz 1024

Tham số hữu ích:
    --conf 0.15   Giảm ngưỡng confidence nếu muốn ưu tiên recall (đỡ bỏ sót chữ ký),
                  đánh đổi bằng việc có thể xuất hiện thêm box sai (false positive).
    --imgsz 1024  Phải khớp (hoặc gần khớp) với imgsz lúc train để độ chính xác tốt nhất.
"""

import argparse
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

CLASS_COLORS = {0: (0, 102, 255), 1: (255, 40, 40)}   # signature=xanh, stamp=do (RGB)
CLASS_NAMES = {0: "signature", 1: "stamp"}


def draw_predictions(image_bgr, result, conf_threshold):
    """Vẽ box + label lên ảnh (dùng OpenCV, làm việc trên bản sao để không sửa ảnh gốc)."""
    img = image_bgr.copy()
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return img, 0

    n_drawn = 0
    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color_rgb = CLASS_COLORS.get(cls_id, (0, 255, 0))
        color_bgr = color_rgb[::-1]  # OpenCV dùng BGR
        name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 3)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), color_bgr, -1)
        cv2.putText(img, label, (x1 + 3, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        n_drawn += 1

    return img, n_drawn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Duong dan file best.pt")
    parser.add_argument("--image", type=str, required=True, help="Duong dan anh can kiem tra")
    parser.add_argument("--conf", type=float, default=0.25, help="Nguong confidence")
    parser.add_argument("--imgsz", type=int, default=1024, help="Kich thuoc input, nen khop luc train")
    parser.add_argument("--save", type=str, default="prediction_result.jpg",
                          help="Duong dan luu anh ket qua")
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.predict(source=args.image, conf=args.conf, imgsz=args.imgsz, verbose=False)
    result = results[0]

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {args.image}")

    annotated, n_drawn = draw_predictions(image_bgr, result, args.conf)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # In ra danh sách phát hiện được (kèm toạ độ) để dễ debug/kiểm tra
    print(f"\nPhat hien {n_drawn} doi tuong (conf >= {args.conf}):")
    if result.boxes is not None:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < args.conf:
                continue
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            print(f"  - {name:10s} conf={conf:.3f}  box=({x1},{y1})-({x2},{y2})")

    # Hien thi bang matplotlib
    plt.figure(figsize=(10, 14))
    plt.imshow(annotated_rgb)
    plt.axis("off")
    plt.title(f"{args.image} — {n_drawn} phat hien (conf >= {args.conf})")
    plt.tight_layout()
    plt.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"\nDa luu anh ket qua: {args.save}")
    plt.show()


if __name__ == "__main__":
    main()
