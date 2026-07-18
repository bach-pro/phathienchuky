"""
generate_dataset.py
====================
Sinh dữ liệu tổng hợp (synthetic) cho bài toán phát hiện CHỮ KÝ (signature) và
CON DẤU (stamp) trên ảnh chứng từ (phiếu thu/chi, hoá đơn), có chủ động tạo ra
các trường hợp con dấu đè lên chữ ký (occlusion) để model học tốt case khó này.

Output: ảnhr .jpg + nhãn YOLO fomat (.txt) trong thư mục dataset/images, dataset/labels
Class id: 0 = signature, 1 = stamp

CÁCH DÙNG NHANH:
    800x1100, 1000x1400, 1200x1600, 1600x1000, 1920x1080
Nếu muốn sinh đúng một kích thước cũ:
    python generate_dataset.py --num_samples 2000 --out_dir dataset_documents_multisig --documents_dir documents --signatures_dir signatures_dir --regions_json document_regions.json --signature_scale 2.8125 --stamp_scale 2.0 --min_signatures 2 --max_signatures 5 --size_profile fixed --img_w 1191 --img_h 1684

CHUẨN BỊ DỮ LIỆU ĐẦU VÀO (khuyến nghị để chất lượng thật hơn):
    - Để các ảnh phiếu/đơn mẫu .png/.jpg vào thư mục --documents_dir. Nếu có,
      script sẽ dùng các ảnh này làm nền thật thay vì tự vẽ form giả.
    - Có thể khai báo vùng đặt chữ ký/con dấu theo từng file document bằng
      --regions_json. Tọa độ vùng là normalized [x1,y1,x2,y2] trong khoảng 0..1.
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
import json

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps


SIGNATURE_INK_COLORS = [
    (5, 30, 85),       # xanh navy rat dam
    (0, 45, 115),      # xanh but bi dam
    (5, 60, 140),      # xanh dam vua
    (20, 80, 165),     # xanh vua
    (45, 110, 190),    # xanh nhat
]

DEFAULT_SIGNATURE_ZONES = [
    [0.08, 0.80, 0.42, 0.96],
    [0.55, 0.80, 0.94, 0.96],
]

DEFAULT_REALISTIC_SIZES = [
    (800, 1100),
    (1000, 1400),
    (1200, 1600),
    (1600, 1000),
    (1920, 1080),
]


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


def validate_normalized_zones(zones, source_name, field_name):
    """Kiem tra va chuan hoa danh sach box normalized [x1,y1,x2,y2]."""
    valid = []
    if zones is None:
        return valid
    if not isinstance(zones, list):
        print(f"[warn] {source_name}.{field_name} phai la list -> bo qua")
        return valid

    for idx, zone in enumerate(zones):
        if not isinstance(zone, list) or len(zone) != 4:
            print(f"[warn] {source_name}.{field_name}[{idx}] khong dung [x1,y1,x2,y2] -> bo qua")
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in zone]
        except (TypeError, ValueError):
            print(f"[warn] {source_name}.{field_name}[{idx}] co gia tri khong phai so -> bo qua")
            continue
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            print(f"[warn] {source_name}.{field_name}[{idx}] phai nam trong khoang 0..1 va x1<x2,y1<y2 -> bo qua")
            continue
        valid.append([x1, y1, x2, y2])
    return valid


def normalize_region_block(block, source_name):
    """Lay signature_zones/stamp_zones hop le tu 1 block cau hinh."""
    if not isinstance(block, dict):
        print(f"[warn] {source_name} phai la object JSON -> bo qua")
        return {}

    normalized = {}
    for field_name in ("signature_zones", "stamp_zones"):
        if field_name not in block:
            continue
        zones = validate_normalized_zones(block.get(field_name), source_name, field_name)
        if zones or block.get(field_name) == []:
            normalized[field_name] = zones
    return normalized


def load_document_region_config(regions_json):
    """
    Doc cau hinh gioi han vung dat chu ky/moc theo tung document.
    Toa do la normalized theo document goc: [x1, y1, x2, y2] trong khoang 0..1.
    """
    with open(regions_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("--regions_json phai la object JSON")

    config = {
        "default": normalize_region_block(raw.get("default", {}), "default"),
        "documents": {},
    }

    documents_raw = raw.get("documents", {})
    if not isinstance(documents_raw, dict):
        print("[warn] documents trong regions_json phai la object -> bo qua")
        documents_raw = {}

    for doc_name, block in documents_raw.items():
        normalized = normalize_region_block(block, f"documents.{doc_name}")
        if normalized:
            config["documents"][os.path.basename(doc_name)] = normalized

    return config


def get_regions_for_document(path, region_config):
    """Ghep default + cau hinh rieng theo basename document."""
    regions = {
        "signature_zones": DEFAULT_SIGNATURE_ZONES,
    }
    if region_config:
        regions.update(region_config.get("default", {}))
        regions.update(region_config.get("documents", {}).get(os.path.basename(path), {}))
    if not regions.get("signature_zones"):
        regions["signature_zones"] = DEFAULT_SIGNATURE_ZONES
    return regions


def normalized_zones_to_pixel_zones(zones, content_box):
    """Doi cac box normalized theo document content thanh box pixel tren canvas output."""
    cx1, cy1, cx2, cy2 = content_box
    content_w = cx2 - cx1
    content_h = cy2 - cy1
    pixel_zones = []
    for x1, y1, x2, y2 in zones:
        pixel_zones.append((
            int(cx1 + content_w * x1),
            int(cy1 + content_h * y1),
            int(cx1 + content_w * x2),
            int(cy1 + content_h * y2),
        ))
    return pixel_zones


def parse_image_sizes(raw):
    """Parse chuoi kich thuoc dang '800x1100,1600x1000'."""
    sizes = []
    for item in raw.split(","):
        item = item.strip().lower().replace(" ", "")
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Kich thuoc khong dung dinh dang WxH: {item}")
        w_text, h_text = item.split("x", 1)
        try:
            w = int(w_text)
            h = int(h_text)
        except ValueError as exc:
            raise ValueError(f"Kich thuoc khong phai so nguyen: {item}") from exc
        if w <= 0 or h <= 0:
            raise ValueError(f"Kich thuoc phai > 0: {item}")
        sizes.append((w, h))

    if not sizes:
        raise ValueError("--img_sizes khong co kich thuoc hop le")
    return sizes


def fit_asset_to_zone(asset, zone, max_fill=0.95):
    """Thu nho asset neu sau khi rotate no lon hon vung dat."""
    zx1, zy1, zx2, zy2 = zone
    zone_w = max(1, zx2 - zx1)
    zone_h = max(1, zy2 - zy1)
    max_w = max(1, int(zone_w * max_fill))
    max_h = max(1, int(zone_h * max_fill))
    scale = min(1.0, max_w / max(1, asset.width), max_h / max(1, asset.height))
    if scale >= 1.0:
        return asset
    new_size = (max(1, int(asset.width * scale)), max(1, int(asset.height * scale)))
    return asset.resize(new_size, Image.Resampling.LANCZOS)


def choose_position_in_zone(asset, zone, img_w, img_h, center_hint=None):
    """Chon toa do dat asset trong 1 zone, co the uu tien quanh center_hint."""
    zx1, zy1, zx2, zy2 = zone
    min_x, min_y = max(zx1, 0), max(zy1, 0)
    max_x = min(zx2, img_w) - asset.width
    max_y = min(zy2, img_h) - asset.height

    if center_hint:
        sx = int(center_hint[0] - asset.width / 2)
        sy = int(center_hint[1] - asset.height / 2)
    else:
        sx = random.randint(min_x, max_x) if max_x > min_x else min_x
        sy = random.randint(min_y, max_y) if max_y > min_y else min_y

    sx = max(0, min(sx, img_w - asset.width))
    sy = max(0, min(sy, img_h - asset.height))
    if max_x > min_x:
        sx = max(min_x, min(sx, max_x))
    if max_y > min_y:
        sy = max(min_y, min(sy, max_y))
    return sx, sy


def load_document_backgrounds(documents_dir, region_config=None):
    """Load cac anh phieu/don mau lam nen that cho synthetic dataset."""
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        paths.extend(glob.glob(os.path.join(documents_dir, ext)))
    paths = sorted(paths)

    documents = []
    for p in paths:
        try:
            im = Image.open(p)
            im = ImageOps.exif_transpose(im).convert("RGB")
            documents.append({
                "path": p,
                "image": im,
                "regions": get_regions_for_document(p, region_config),
            })
        except Exception as e:
            print(f"[warn] Khong doc duoc document {p}: {e}")
    return documents


def prepare_document_background(document_img, width, height):
    """
    Dua anh phieu/don mau vao canvas output ma van giu ty le.
    Tra ve anh nen va bbox noi dung document tren canvas de dat chu ky dung vung.
    """
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    src_w, src_h = document_img.size
    if src_w <= 0 or src_h <= 0:
        return canvas, (0, 0, width, height)

    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = document_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas, (x, y, x + new_w, y + new_h)


# ----------------------------------------------------------------------------
# 2. SINH CHỮ KÝ GIẢ (fallback khi chưa có dataset chữ ký thật)
# ----------------------------------------------------------------------------

def _bezier_point(p0, p1, p2, p3, t):
    x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
    y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
    return (x, y)


def random_signature_ink_color(alpha=255):
    """Chon muc xanh lam voi nhieu muc do dam/nhat khac nhau."""
    r, g, b = random.choice(SIGNATURE_INK_COLORS)
    # Jitter dong deu giup da dang sac do ma van luon giu mau xanh lam.
    jitter = random.randint(-12, 12)
    r = max(0, min(255, r + jitter))
    g = max(0, min(255, g + jitter))
    b = max(0, min(255, b + jitter))
    return (r, g, b, alpha)


def recolor_signature_ink(sig_img, color=None):
    """
    Doi mau net muc trong anh chu ky RGBA, giu lai alpha va do dam nhat tu anh goc.
    Ham nay dung cho chu ky that de train khong bi lech khi test gap chu ky mau xanh.
    """
    sig_img = sig_img.convert("RGBA")
    if color is None:
        color = random_signature_ink_color(alpha=255)

    arr = np.array(sig_img).astype(np.float32)
    alpha = arr[..., 3]
    if alpha.max() <= 0:
        return sig_img

    rgb = arr[..., :3]
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    darkness = 1.0 - gray / 255.0
    darkness = np.clip(darkness * 1.18, 0.35, 1.0)

    ink = np.array(color[:3], dtype=np.float32)
    paper = np.array([255, 255, 255], dtype=np.float32)
    recolored_rgb = paper * (1.0 - darkness[..., None]) + ink * darkness[..., None]

    mask = alpha > 0
    arr[..., :3][mask] = recolored_rgb[mask]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGBA")


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
        color = random_signature_ink_color(alpha=255)

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
    red = (165, 0, 0, 225)  # mau muc dau dam hon mot chut, van giu do trong tu nhien

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


def compose_sample(
    bg_size=(1000, 1400),
    signatures_pool=None,
    documents_pool=None,
    signature_scale=2.8125,
    stamp_scale=2.0,
    min_signatures=2,
    max_signatures=5,
):
    """
    Sinh 1 sample hoàn chỉnh: nền + (có thể có) chữ ký + (có thể có) con dấu,
    trả về ảnh PIL và list nhãn [(class_id, x1,y1,x2,y2), ...] theo pixel.

    class_id: 0 = signature, 1 = stamp
    """
    W, H = bg_size
    document_regions = None
    if documents_pool:
        document = random.choice(documents_pool)
        document_img = document["image"]
        document_regions = document.get("regions")
        img, content_box = prepare_document_background(document_img, W, H)
    else:
        img = generate_background(W, H)
        content_box = (0, 0, W, H)
    labels = []

    # Vung dat chu ky/moc co the cau hinh rieng theo tung document bang --regions_json.
    signature_zone_defs = DEFAULT_SIGNATURE_ZONES
    stamp_zone_defs = None
    stamp_allowed = True
    if document_regions:
        signature_zone_defs = document_regions.get("signature_zones") or DEFAULT_SIGNATURE_ZONES
        if "stamp_zones" in document_regions:
            stamp_zone_defs = document_regions.get("stamp_zones")
            stamp_allowed = bool(stamp_zone_defs)

    zones = normalized_zones_to_pixel_zones(signature_zone_defs, content_box)
    if not zones:
        zones = normalized_zones_to_pixel_zones(DEFAULT_SIGNATURE_ZONES, content_box)
    stamp_zones = normalized_zones_to_pixel_zones(stamp_zone_defs, content_box) if stamp_zone_defs else []

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

    if scenario == "empty":
        pass

    else:
        stamp_zone = None
        if scenario in ("sig_and_stamp_separate", "sig_and_stamp_overlap") and stamp_allowed and stamp_zones:
            stamp_zone = random.choice(stamp_zones)

        if len(zones) <= 1:
            signature_count = 1
        else:
            min_count = min(max(1, min_signatures), len(zones))
            max_count = min(max(min_count, max_signatures), len(zones))
            signature_count = random.randint(min_count, max_count)

        selected_zones = random.sample(zones, signature_count)
        if stamp_zone:
            stamp_cx = (stamp_zone[0] + stamp_zone[2]) / 2
            stamp_cy = (stamp_zone[1] + stamp_zone[3]) / 2
            director_zone = min(
                zones,
                key=lambda z: ((z[0] + z[2]) / 2 - stamp_cx) ** 2 + ((z[1] + z[3]) / 2 - stamp_cy) ** 2,
            )
            if director_zone not in selected_zones:
                selected_zones[0] = director_zone

        sig_boxes = []
        for zone in selected_zones:
            zx1, zy1, zx2, zy2 = zone

            # --- Lay 1 chu ky (that neu co, khong thi sinh gia) ---
            if signatures_pool:
                sig = random.choice(signatures_pool).copy()
            else:
                sig = generate_fake_signature(
                    width=random.randint(220, 320), height=random.randint(80, 130)
                )
            sig = recolor_signature_ink(sig)

            # Chuan hoa kich thuoc chu ky theo ty le vung ky, khong phu thuoc
            # do phan giai goc cua anh chu ky upload vao signatures_dir.
            zone_w = zx2 - zx1
            signature_size_bucket = random.choice([
                (0.18, 0.28),  # nho
                (0.28, 0.40),  # vua
                (0.40, 0.55),  # lon
            ])
            target_w = zone_w * random.uniform(*signature_size_bucket) * signature_scale
            scale_factor = target_w / sig.width
            new_w = max(20, int(sig.width * scale_factor))
            new_h = max(10, int(sig.height * scale_factor))
            sig = sig.resize((new_w, new_h))

            angle = random.uniform(-8, 8)
            sig = sig.rotate(angle, expand=True)
            sig = fit_asset_to_zone(sig, zone)

            # Uu tien dat trong vung ky, nhung van kep cung theo bien anh output.
            sx, sy = choose_position_in_zone(sig, zone, W, H)

            sig_opacity = random.uniform(0.88, 1.0)
            sig_box = paste_with_alpha(img, sig, sx, sy, opacity=sig_opacity)
            sig_boxes.append(sig_box)
            labels.append((0,) + sig_box)  # class 0 = signature

        if scenario in ("sig_and_stamp_separate", "sig_and_stamp_overlap") and stamp_allowed:
            # QUAN TRONG: kich thuoc dau cung phai theo TY LE trang/vung ky,
            # khong dung so pixel co dinh (130-190px) -- neu doi img_w/img_h
            # (vd sinh anh 640x896 thay vi 1000x1400) thi dau co dinh se thanh
            # qua to hoac qua nho mot cach khong nhat quan.
            if stamp_zone is None:
                stamp_zone = random.choice(stamp_zones) if stamp_zones else None
            zone_w = selected_zones[0][2] - selected_zones[0][0]
            stamp_ref_w = (stamp_zone[2] - stamp_zone[0]) if stamp_zone else zone_w
            stamp_size_bucket = random.choice([
                (0.22, 0.32),  # nho/nhat
                (0.32, 0.44),  # vua
                (0.44, 0.56),  # lon/de thay
            ])
            stamp_size = int(stamp_ref_w * random.uniform(*stamp_size_bucket) * stamp_scale)
            stamp_size = max(60, stamp_size)  # tranh dau qua nho khi zone hep
            stamp = generate_stamp(size=stamp_size)
            if stamp_zone:
                stamp = fit_asset_to_zone(stamp, stamp_zone, max_fill=0.95)
            stamp_opacity = random.uniform(0.70, 0.92)

            if stamp_zone:
                center_hint = None
                if scenario == "sig_and_stamp_overlap" and sig_boxes:
                    stamp_cx = (stamp_zone[0] + stamp_zone[2]) / 2
                    stamp_cy = (stamp_zone[1] + stamp_zone[3]) / 2
                    nearest_sig_box = min(
                        sig_boxes,
                        key=lambda b: ((b[0] + b[2]) / 2 - stamp_cx) ** 2 + ((b[1] + b[3]) / 2 - stamp_cy) ** 2,
                    )
                    center_hint = (
                        int((nearest_sig_box[0] + nearest_sig_box[2]) / 2),
                        int((nearest_sig_box[1] + nearest_sig_box[3]) / 2),
                    )
                stx, sty = choose_position_in_zone(stamp, stamp_zone, W, H, center_hint=center_hint)
            elif scenario == "sig_and_stamp_overlap":
                # Cố tình đặt tâm con dấu lệch vào vùng chữ ký để tạo occlusion 30-70%
                overlap_ratio = random.uniform(0.3, 0.7)
                target_sig_box = random.choice(sig_boxes)
                cx = int(target_sig_box[0] + (target_sig_box[2] - target_sig_box[0]) * overlap_ratio)
                cy = int((target_sig_box[1] + target_sig_box[3]) / 2)
                stx = cx - stamp.width // 2
                sty = cy - stamp.height // 2
            else:
                # Đặt cách xa chữ ký, không chồng lấn
                target_sig_box = random.choice(sig_boxes)
                stx = target_sig_box[2] + random.randint(10, 40)
                sty = target_sig_box[1] - random.randint(0, 20)
                if stx + stamp.width > W - 20:
                    stx = max(20, target_sig_box[0] - stamp.width - 20)

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

def transform_labels_affine(labels, matrix, img_w, img_h, min_box_size=4):
    """Bien doi bbox theo ma tran affine 2x3, clamp lai trong bien anh."""
    transformed_labels = []
    m = np.asarray(matrix, dtype=np.float32)

    for cls, x1, y1, x2, y2 in labels:
        corners = np.array([
            [x1, y1, 1.0],
            [x2, y1, 1.0],
            [x2, y2, 1.0],
            [x1, y2, 1.0],
        ], dtype=np.float32)
        warped = corners @ m.T
        nx1 = float(np.clip(warped[:, 0].min(), 0, img_w))
        ny1 = float(np.clip(warped[:, 1].min(), 0, img_h))
        nx2 = float(np.clip(warped[:, 0].max(), 0, img_w))
        ny2 = float(np.clip(warped[:, 1].max(), 0, img_h))
        if nx2 - nx1 < min_box_size or ny2 - ny1 < min_box_size:
            continue
        transformed_labels.append((cls, nx1, ny1, nx2, ny2))

    return transformed_labels


def apply_capture_geometry_augmentation(pil_img, labels):
    """
    Mo phong scan/chup bi lech nhe: rotate, translate, scale nho.
    Anh giu nguyen kich thuoc, label duoc bien doi cung anh.
    """
    if random.random() > 0.75:
        return pil_img, labels

    img = np.array(pil_img.convert("RGB"))
    img_h, img_w = img.shape[:2]
    angle = random.uniform(-2.5, 2.5)
    scale = random.uniform(0.96, 1.03)
    tx = random.uniform(-0.025, 0.025) * img_w
    ty = random.uniform(-0.025, 0.025) * img_h

    matrix = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty

    warped = cv2.warpAffine(
        img,
        matrix,
        (img_w, img_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    transformed_labels = transform_labels_affine(labels, matrix, img_w, img_h)
    return Image.fromarray(warped), transformed_labels


def add_uneven_lighting(img):
    """Them gradient sang/toi nhe nhu anh chup dien thoai/scan khong deu den."""
    img_f = img.astype(np.float32)
    h, w = img.shape[:2]
    xs = np.linspace(-1, 1, w, dtype=np.float32)
    ys = np.linspace(-1, 1, h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    angle = random.uniform(0, math.pi)
    direction = math.cos(angle) * grid_x + math.sin(angle) * grid_y
    direction = (direction - direction.min()) / max(1e-6, direction.max() - direction.min())
    strength = random.uniform(0.10, 0.28)
    gradient = 1.0 + (direction - 0.5) * strength
    img_f *= gradient[..., None]
    return np.clip(img_f, 0, 255).astype(np.uint8)


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

    # 4. Anh sang khong deu (dien thoai chup duoi den phong/scan bi lech sang)
    if random.random() < 0.35:
        img = add_uneven_lighting(img)

    out = Image.fromarray(img)

    # 5. Nén JPEG artifact (lưu rồi đọc lại với quality thấp)
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
    parser.add_argument("--documents_dir", type=str, default=None,
                            help="Thư mục chứa ảnh phiếu/đơn mẫu dùng làm nền thật. "
                                "Nếu không cung cấp, script sẽ tự vẽ nền chứng từ giả.")
    parser.add_argument("--regions_json", type=str, default=None,
                            help="File JSON khai báo vùng đặt chữ ký/con dấu theo từng document. "
                                "Tọa độ normalized [x1,y1,x2,y2] theo ảnh document gốc.")
    parser.add_argument("--signature_scale", type=float, default=2.8125,
                            help="He so phong to chu ky so voi kich thuoc mac dinh.")
    parser.add_argument("--stamp_scale", type=float, default=2.0,
                            help="He so phong to con dau/moc so voi kich thuoc mac dinh.")
    parser.add_argument("--min_signatures", type=int, default=2,
                            help="So chu ky toi thieu tren moi document co chu ky.")
    parser.add_argument("--max_signatures", type=int, default=5,
                            help="So chu ky toi da tren moi document co chu ky.")
    parser.add_argument("--img_w", type=int, default=1000)
    parser.add_argument("--img_h", type=int, default=1400)
    parser.add_argument("--size_profile", choices=["fixed", "realistic"], default="realistic",
                            help="'fixed' dung --img_w/--img_h; 'realistic' random nhieu ti le "
                                "portrait/landscape theo DEFAULT_REALISTIC_SIZES.")
    parser.add_argument("--img_sizes", type=str, default=None,
                            help="Danh sach kich thuoc custom dang WxH, cach nhau bang dau phay, "
                                "vd 800x1100,1000x1400,1600x1000. Neu truyen tham so nay "
                                "se uu tien hon --size_profile.")
    parser.add_argument("--no_geometry_aug", action="store_true",
                            help="Tat rotate/translate/scale nhe sau khi ghep object. Dung khi can debug label.")
    parser.add_argument("--seed", type=int, default=None,
                            help="Seed tuy chon de lap lai dataset synthetic.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.signature_scale <= 0:
        raise ValueError("--signature_scale phai > 0")
    if args.stamp_scale <= 0:
        raise ValueError("--stamp_scale phai > 0")
    if args.min_signatures <= 0:
        raise ValueError("--min_signatures phai > 0")
    if args.max_signatures < args.min_signatures:
        raise ValueError("--max_signatures phai >= --min_signatures")
    if args.img_w <= 0 or args.img_h <= 0:
        raise ValueError("--img_w/--img_h phai > 0")

    if args.img_sizes:
        output_sizes = parse_image_sizes(args.img_sizes)
    elif args.size_profile == "realistic":
        output_sizes = DEFAULT_REALISTIC_SIZES
    else:
        output_sizes = [(args.img_w, args.img_h)]

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

    region_config = None
    if args.regions_json:
        if os.path.isfile(args.regions_json):
            region_config = load_document_region_config(args.regions_json)
            print(f"Da load cau hinh vung dat chu ky/moc tu {args.regions_json}")
        else:
            raise FileNotFoundError(f"Khong tim thay --regions_json: {args.regions_json}")

    documents_pool = None
    if args.documents_dir and os.path.isdir(args.documents_dir):
        documents_pool = load_document_backgrounds(args.documents_dir, region_config=region_config)
        print(f"Da load {len(documents_pool)} phieu/don mau tu {args.documents_dir}")
        if not documents_pool:
            print("[warn] Khong load duoc document nao -> se tu ve nen chung tu gia.")
            documents_pool = None
    else:
        print("Khong co --documents_dir hop le -> se tu ve nen chung tu gia.")

    print("Kich thuoc output se random trong:", ", ".join(f"{w}x{h}" for w, h in output_sizes))

    for i in range(args.num_samples):
        img_w, img_h = random.choice(output_sizes)
        img, labels = compose_sample(
            bg_size=(img_w, img_h),
            signatures_pool=signatures_pool,
            documents_pool=documents_pool,
            signature_scale=args.signature_scale,
            stamp_scale=args.stamp_scale,
            min_signatures=args.min_signatures,
            max_signatures=args.max_signatures,
        )
        if not args.no_geometry_aug:
            img, labels = apply_capture_geometry_augmentation(img, labels)
        img = apply_scan_augmentation(img)

        fname = f"sample_{i:05d}"
        img.save(os.path.join(img_dir, fname + ".jpg"), quality=90)
        save_yolo_label(labels, img_w, img_h, os.path.join(lbl_dir, fname + ".txt"))

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
    print(f"  python train.py --data {args.out_dir}_split/data.yaml --model yolo11s.pt --imgsz 1024 --epochs 100")


if __name__ == "__main__":
    main()
