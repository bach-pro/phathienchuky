"""
generate_dataset.py
====================
Sinh dữ liệu tổng hợp (synthetic) cho bài toán phát hiện CHỮ KÝ (signature) và
CON DẤU (stamp) trên ảnh chứng từ (phiếu thu/chi, hoá đơn), có chủ động tạo ra
các trường hợp con dấu đè lên chữ ký (occlusion) để model học tốt case khó này.

Output: ảnhr .jpg + nhãn YOLO fomat (.txt) trong thư mục dataset/images, dataset/labels
Class id: 0 = signature, 1 = stamp

CÁCH DÙNG NHANH:
    python generate_dataset.py --num_samples 2000 --out_dir dataset --signatures_dir signatures_dir

CHUẨN BỊ DỮ LIỆU ĐẦU VÀO (khuyến nghị để chất lượng thật hơn):
    - Tải chữ ký thật từ CEDAR / GPDS / ICDAR SigComp, để các ảnh .png/.jpg
      (nền trắng, mực đen/xanh) vào thư mục --signatures_dir
    - Nếu KHÔNG có, script tự sinh chữ ký giả bằng đường cong Bezier ngẫu nhiên
      (chất lượng thấp hơn thật nhưng vẫn giúp model học được hình dạng/pattern cơ bản)
    - Con dấu được sinh HOÀN TOÀN bằng code (không cần dataset), vì con dấu có
      cấu trúc hình học đều đặn (tròn/vuông + viền + chữ) dễ mô phỏng.
"""

import os
import random
import math
import argparse
import glob

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# ----------------------------------------------------------------------------
# 1. SINH NỀN TÀI LIỆU (background giống phiếu thu/chi/hoá đơn)
# ----------------------------------------------------------------------------

def generate_background(width, height):
    """Sinh 1 ảnh nền trắng giống chứng từ: có bảng, đường kẻ, vài dòng text giả."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()

    # Tiêu đề giả
    titles = ["PHIẾU CHI", "PHIẾU THU", "HOA DON GTGT", "BIEN BAN BAN GIAO", "PHIEU THANH TOAN"]
    draw.text((width * 0.32, 30), random.choice(titles), fill=(0, 0, 0), font=font_title)

    # Vài dòng thông tin giả phía trên
    fake_lines = [
        "So chung tu: {}".format(random.randint(1000, 9999)),
        "Ngay lap: {:02d}/{:02d}/2026".format(random.randint(1, 28), random.randint(1, 12)),
        "Don vi: Cong ty TNHH ABC",
        "Nguoi nop/nhan: Nguyen Van {}".format(random.choice("ABCDEFGH")),
        "So tien: {:,} VND".format(random.randint(100000, 50000000)),
    ]
    y = 80
    for line in fake_lines:
        draw.text((60, y), line, fill=(20, 20, 20), font=font_text)
        y += 26

    # Vẽ bảng kẻ ô giả (mô phỏng bảng chi tiết)
    table_top = y + 20
    table_bottom = int(height * 0.65)
    n_rows = random.randint(3, 6)
    row_h = (table_bottom - table_top) // n_rows
    for r in range(n_rows + 1):
        yy = table_top + r * row_h
        draw.line([(60, yy), (width - 60, yy)], fill=(0, 0, 0), width=1)
    for xx in [60, width * 0.6, width - 60]:
        draw.line([(xx, table_top), (xx, table_bottom)], fill=(0, 0, 0), width=1)

    # Khu vực chữ ký ở cuối trang (label gợi ý, không phải box nhãn)
    sign_zone_y = int(height * 0.78)
    draw.text((80, sign_zone_y), "Nguoi lap phieu", fill=(0, 0, 0), font=font_text)
    draw.text((width - 260, sign_zone_y), "Nguoi ky duyet", fill=(0, 0, 0), font=font_text)
    draw.text((80, sign_zone_y + 20), "(Ky, ghi ro ho ten)", fill=(90, 90, 90), font=font_text)
    draw.text((width - 260, sign_zone_y + 20), "(Ky, ghi ro ho ten)", fill=(90, 90, 90), font=font_text)

    return img


# ----------------------------------------------------------------------------
# 2. SINH CHỮ KÝ GIẢ (fallback khi chưa có dataset chữ ký thật)
# ----------------------------------------------------------------------------

def _bezier_point(p0, p1, p2, p3, t):
    x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
    y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
    return (x, y)


def generate_fake_signature(width=300, height=120, color=None):
    """
    Sinh chữ ký giả bằng nhiều đường cong Bezier nối tiếp nhau, mô phỏng nét bút.
    Chỉ nên dùng làm dữ liệu bổ sung/tạm thời -- ưu tiên dùng chữ ký thật (CEDAR/GPDS)
    nếu có, vì chữ ký thật có texture nét bút (độ đậm nhạt, tốc độ nét) mà cách sinh
    này không mô phỏng được đầy đủ.
    """
    img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    if color is None:
        # Màu mực phổ biến: xanh dương đậm hoặc đen
        color = random.choice([(20, 30, 120, 255), (10, 10, 10, 255), (0, 40, 90, 255)])

    n_strokes = random.randint(2, 4)
    cursor_x = width * 0.05
    for _ in range(n_strokes):
        stroke_len = random.uniform(width * 0.25, width * 0.5)
        p0 = (cursor_x, random.uniform(height * 0.3, height * 0.7))
        p1 = (cursor_x + stroke_len * 0.3, random.uniform(0, height))
        p2 = (cursor_x + stroke_len * 0.6, random.uniform(0, height))
        p3 = (cursor_x + stroke_len, random.uniform(height * 0.3, height * 0.7))

        pts = [_bezier_point(p0, p1, p2, p3, t) for t in np.linspace(0, 1, 60)]
        line_w = random.randint(2, 4)
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=color, width=line_w)

        cursor_x += stroke_len * random.uniform(0.7, 0.95)

    # Vài nét gạch chân / dấu chấm ngẫu nhiên cho giống chữ ký thật hơn
    if random.random() < 0.5:
        yy = height * random.uniform(0.75, 0.9)
        draw.line([(width * 0.05, yy), (width * 0.7, yy)], fill=color, width=2)

    img = img.filter(ImageFilter.GaussianBlur(0.4))
    return img


def load_real_signatures(signatures_dir):
    """Load các ảnh chữ ký thật (đã tải sẵn từ CEDAR/GPDS...) và tách nền trắng -> alpha."""
    paths = glob.glob(os.path.join(signatures_dir, "*.png")) + \
        glob.glob(os.path.join(signatures_dir, "*.jpg")) + \
        glob.glob(os.path.join(signatures_dir, "*.jpeg"))
    signatures = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
            arr = np.array(im)
            # Coi pixel gần trắng là nền -> trong suốt
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            alpha = np.where(gray > 235, 0, 255).astype(np.uint8)
            rgba = np.dstack([arr, alpha])
            signatures.append(Image.fromarray(rgba, mode="RGBA"))
        except Exception as e:
            print(f"[warn] Khong doc duoc {p}: {e}")
    return signatures


# ----------------------------------------------------------------------------
# 3. SINH CON DẤU (procedural, không cần dataset)
# ----------------------------------------------------------------------------

def generate_stamp(size=180):
    """Sinh con dấu tròn màu đỏ kiểu công ty/kho bạc, có viền + chữ vòng cung + ngôi sao/text giữa."""
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    red = (200, 20, 20, 200)  # màu mực dấu, hơi trong suốt như dấu thật đóng nhạt

    margin = 8
    draw.ellipse([margin, margin, size - margin, size - margin], outline=red, width=4)
    draw.ellipse([margin + 14, margin + 14, size - margin - 14, size - margin - 14], outline=red, width=2)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    center = size / 2
    text_center = random.choice(["CTY ABC", "KE TOAN", "DA THU", "PHONG TC-KT"])
    bbox = draw.textbbox((0, 0), text_center, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((center - tw / 2, center - th / 2), text_center, fill=red, font=font)

    # Chữ cong theo viền trên (mô phỏng tên công ty chạy vòng cung)
    arc_text = "CONG TY TNHH THUONG MAI"
    radius = size / 2 - 26
    angle_start = 200
    angle_step = 10
    for i, ch in enumerate(arc_text[:16]):
        angle = math.radians(angle_start - i * angle_step)
        x = center + radius * math.cos(angle)
        y = center + radius * math.sin(angle)
        char_img = Image.new("RGBA", (20, 20), (255, 255, 255, 0))
        cd = ImageDraw.Draw(char_img)
        cd.text((2, 2), ch, fill=red, font=font)
        rot_angle = -(angle_start - i * angle_step) - 90
        char_img = char_img.rotate(rot_angle, expand=True)
        img.paste(char_img, (int(x - char_img.width / 2), int(y - char_img.height / 2)), char_img)

    # Làm mực hơi loang/nhạt không đều như dấu đóng tay thật
    arr = np.array(img)
    noise = np.random.normal(0, 12, arr[..., 3].shape).astype(np.int16)
    alpha = arr[..., 3].astype(np.int16) + noise
    arr[..., 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGBA")
    img = img.rotate(random.uniform(-15, 15), expand=True)
    return img


# ----------------------------------------------------------------------------
# 4. GHÉP (COMPOSITE) SIGNATURE + STAMP LÊN NỀN, CÓ CHO PHÉP CHỒNG LẤN
# ----------------------------------------------------------------------------

def paste_with_alpha(bg, fg, x, y, opacity=1.0):
    """Dán ảnh fg (RGBA) lên bg tại (x,y) với độ mờ opacity, trả về bbox (x1,y1,x2,y2)."""
    fg = fg.copy()
    if opacity < 1.0:
        alpha = fg.split()[3].point(lambda p: int(p * opacity))
        fg.putalpha(alpha)
    bg.paste(fg, (x, y), fg)
    return (x, y, x + fg.width, y + fg.height)


def compose_sample(bg_size=(1000, 1400), signatures_pool=None):
    """
    Sinh 1 sample hoàn chỉnh: nền + (có thể có) chữ ký + (có thể có) con dấu,
    trả về ảnh PIL và list nhãn [(class_id, x1,y1,x2,y2), ...] theo pixel.

    class_id: 0 = signature, 1 = stamp
    """
    W, H = bg_size
    img = generate_background(W, H)
    labels = []

    # Vùng hợp lý để đặt chữ ký/dấu: 2 khu vực cuối trang (bên trái và bên phải)
    zones = [
        (60, int(H * 0.80), int(W * 0.42), int(H * 0.96)),
        (int(W * 0.55), int(H * 0.80), W - 60, int(H * 0.96)),
    ]

    # Quyết định kịch bản cho sample này
    r = random.random()
    if r < 0.10:
        scenario = "empty"          # không có gì (negative sample -- quan trọng để giảm false positive)
    elif r < 0.45:
        scenario = "sig_only"
    elif r < 0.70:
        scenario = "sig_and_stamp_separate"
    else:
        scenario = "sig_and_stamp_overlap"   # trường hợp khó: dấu đè lên chữ ký

    zone = random.choice(zones)
    zx1, zy1, zx2, zy2 = zone

    if scenario == "empty":
        pass

    else:
        # --- Lấy 1 chữ ký (thật nếu có, không thì sinh giả) ---
        if signatures_pool:
            sig = random.choice(signatures_pool).copy()
        else:
            sig = generate_fake_signature(
                width=random.randint(220, 320), height=random.randint(80, 130)
            )

        # QUAN TRỌNG: chuẩn hoá kích thước chữ ký theo TỶ LỆ vùng ký (zone),
        # không phụ thuộc độ phân giải gốc của ảnh chữ ký (đặc biệt là ảnh thật
        # upload vào signatures_dir có thể có độ phân giải rất khác nhau).
        # Nếu resize chỉ theo % kích thước gốc (như code cũ) thì ảnh chữ ký
        # gốc lớn (vd 2000x800px) sẽ cho ra chữ ký to bất thường so với trang.
        # Ty le thuc te: chu ky tay thuong chi chiem ~25-40% chieu rong vung ky
        # (vd o A4 that, chu ky rong ~3-5cm trong khi vung ky rong ~10-15cm).
        # Ty le cu (0.45-0.75) qua cao lam chu ky trong "to bat thuong" so voi trang.
        zone_w = zx2 - zx1
        target_w = zone_w * random.uniform(0.25, 0.40)
        scale_factor = target_w / sig.width
        new_w = max(20, int(sig.width * scale_factor))
        new_h = max(10, int(sig.height * scale_factor))
        sig = sig.resize((new_w, new_h))

        angle = random.uniform(-8, 8)
        sig = sig.rotate(angle, expand=True)

        # Ưu tiên đặt trong vùng ký (zone), nhưng LUÔN kẹp cứng theo biên trang
        # thật (0..W, 0..H) để chữ ký không bao giờ tràn ra ngoài ảnh, kể cả khi
        # sau khi rotate(expand=True) kích thước tăng lên hơn dự tính.
        min_x, min_y = max(zx1, 0), max(zy1, 0)
        max_x = min(zx2, W) - sig.width
        max_y = min(zy2, H) - sig.height
        sx = random.randint(min_x, max_x) if max_x > min_x else min_x
        sy = random.randint(min_y, max_y) if max_y > min_y else min_y
        sx = max(0, min(sx, W - sig.width))
        sy = max(0, min(sy, H - sig.height))

        sig_opacity = random.uniform(0.75, 1.0)
        sig_box = paste_with_alpha(img, sig, sx, sy, opacity=sig_opacity)
        labels.append((0,) + sig_box)  # class 0 = signature

        if scenario in ("sig_and_stamp_separate", "sig_and_stamp_overlap"):
            # QUAN TRONG: kich thuoc dau cung phai theo TY LE trang/vung ky,
            # khong dung so pixel co dinh (130-190px) -- neu doi img_w/img_h
            # (vd sinh anh 640x896 thay vi 1000x1400) thi dau co dinh se thanh
            # qua to hoac qua nho mot cach khong nhat quan.
            stamp_size = int(zone_w * random.uniform(0.28, 0.42))
            stamp_size = max(60, stamp_size)  # tranh dau qua nho khi zone hep
            stamp = generate_stamp(size=stamp_size)
            stamp_opacity = random.uniform(0.55, 0.85)

            if scenario == "sig_and_stamp_overlap":
                # Cố tình đặt tâm con dấu lệch vào vùng chữ ký để tạo occlusion 30-70%
                overlap_ratio = random.uniform(0.3, 0.7)
                cx = int(sig_box[0] + (sig_box[2] - sig_box[0]) * overlap_ratio)
                cy = int((sig_box[1] + sig_box[3]) / 2)
                stx = cx - stamp.width // 2
                sty = cy - stamp.height // 2
            else:
                # Đặt cách xa chữ ký, không chồng lấn
                stx = sig_box[2] + random.randint(10, 40)
                sty = sig_box[1] - random.randint(0, 20)
                if stx + stamp.width > W - 20:
                    stx = max(20, sig_box[0] - stamp.width - 20)

            stx = max(0, min(stx, W - stamp.width))
            sty = max(0, min(sty, H - stamp.height))

            stamp_box = paste_with_alpha(img, stamp, stx, sty, opacity=stamp_opacity)
            labels.append((1,) + stamp_box)  # class 1 = stamp
            # Lưu ý quan trọng: box của signature GIỮ NGUYÊN như lúc dán,
            # KHÔNG cắt bớt phần bị dấu che -- vì ground truth phản ánh
            # vùng chữ ký thật sự tồn tại, kể cả khi bị che một phần.

    return img, labels


# ----------------------------------------------------------------------------
# 5. AUGMENTATION MÔ PHỎNG ẢNH SCAN THẬT
# ----------------------------------------------------------------------------

def apply_scan_augmentation(pil_img):
    """
    Mô phỏng artifact scan/photocopy không làm thay đổi hình học ảnh.

    Lưu ý: bbox đã được tạo trước khi gọi hàm này. Vì vậy không xoay/skew/crop
    toàn ảnh ở đây, nếu không nhãn YOLO sẽ bị lệch. Các augmentation hình học
    nên để Ultralytics xử lý trong train.py vì thư viện sẽ biến đổi bbox cùng ảnh.
    """
    img = np.array(pil_img.convert("RGB"))

    # 1. Nhiễu hạt kiểu scan
    if random.random() < 0.6:
        noise = np.random.normal(0, random.uniform(3, 10), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 2. Làm mờ nhẹ (mô phỏng photocopy/out of focus)
    if random.random() < 0.4:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # 3. Giảm tương phản nhẹ / ánh sáng không đều (mô phỏng scan ám vàng, thiếu sáng)
    if random.random() < 0.3:
        alpha = random.uniform(0.85, 1.05)
        beta = random.uniform(-10, 10)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    out = Image.fromarray(img)

    # 4. Nén JPEG artifact (lưu rồi đọc lại với quality thấp)
    if random.random() < 0.5:
        import io
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=random.randint(45, 80))
        buf.seek(0)
        out = Image.open(buf).convert("RGB")

    return out


# ----------------------------------------------------------------------------
# 6. XUẤT FORMAT YOLO
# ----------------------------------------------------------------------------

def save_yolo_label(labels, img_w, img_h, label_path):
    lines = []
    for cls, x1, y1, x2, y2 in labels:
        x1, x2 = max(0, x1), min(img_w, x2)
        y1, y2 = max(0, y1), min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(label_path, "w") as f:
        f.write("\n".join(lines))


# ----------------------------------------------------------------------------
# 7. MAIN
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--out_dir", type=str, default="dataset")
    parser.add_argument("--signatures_dir", type=str, default=None,
                            help="Thư mục chứa ảnh chữ ký thật đã tải (CEDAR/GPDS...). "
                                "Nếu không cung cấp, script sẽ tự sinh chữ ký giả.")
    parser.add_argument("--img_w", type=int, default=1000)
    parser.add_argument("--img_h", type=int, default=1400)
    args = parser.parse_args()

    img_dir = os.path.join(args.out_dir, "images")
    lbl_dir = os.path.join(args.out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    signatures_pool = None
    if args.signatures_dir and os.path.isdir(args.signatures_dir):
        signatures_pool = load_real_signatures(args.signatures_dir)
        print(f"Da load {len(signatures_pool)} chu ky that tu {args.signatures_dir}")
    else:
        print("Khong co --signatures_dir hop le -> se tu sinh chu ky gia (Bezier).")

    for i in range(args.num_samples):
        img, labels = compose_sample(bg_size=(args.img_w, args.img_h), signatures_pool=signatures_pool)
        img = apply_scan_augmentation(img)

        fname = f"sample_{i:05d}"
        img.save(os.path.join(img_dir, fname + ".jpg"), quality=90)
        save_yolo_label(labels, args.img_w, args.img_h, os.path.join(lbl_dir, fname + ".txt"))

        if (i + 1) % 100 == 0:
            print(f"Da sinh {i + 1}/{args.num_samples} samples")

    # Ghi file YAML tham khảo cho dataset CHƯA split. Không dùng file này để train
    # vì train/val sẽ trỏ cùng thư mục và làm metric validation bị ảo.
    yaml_path = os.path.join(args.out_dir, "data_unsplit_DO_NOT_TRAIN.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            "# Dataset chua split: KHONG dung file nay de train/validate.\n"
            "# Hay chay split_dataset.py de tao data.yaml rieng cho train/val.\n"
            f"path: {os.path.abspath(args.out_dir)}\n"
            f"train: images\n"
            f"val: images\n"
            f"names:\n  0: signature\n  1: stamp\n"
        )

    print(f"\nHoan tat. Dataset tai: {os.path.abspath(args.out_dir)}")
    print(f"File tham khao dataset chua split: {yaml_path}")
    print("Buoc tiep theo nen lam:")
    print(f"  python split_dataset.py --src_dir {args.out_dir} --dst_dir {args.out_dir}_split --val_ratio 0.15")
    print("Sau do train voi file data.yaml da split, vi du:")
    print(f"  yolo detect train data={args.out_dir}_split/data.yaml model=yolov8n.pt imgsz=1024 epochs=50")


if __name__ == "__main__":
    main()
