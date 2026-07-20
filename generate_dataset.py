"""
Sinh dataset YOLO phat hien chu ky va con dau tu hai nguon du lieu:

1. ``signatures_dir``: anh chu ky that, co the co nen trang/xam/mau va chu in thua.
2. ``documents``: anh bieu mau sach; vung ky/dong dau nam trong
   ``document_regions.json``.

Mac dinh script:

- nap truc tiep ``signatures_dir``, ``documents`` va ``document_regions.json``;
- lam sach, tach nen va crop sat net muc cua tung chu ky;
- dung moi document gan nhu deu nhau thay vi random lech phan bo;
- giu nguyen chieu va ty le goc cua document (portrait/landscape);
- xuat anh, nhan YOLO va ``manifest.jsonl`` de truy vet nguon mau.

Vi du:

    python generate_dataset.py --num_samples 2000 --out_dir dataset_generated

Kiem tra dau vao ma khong sinh file:

    python generate_dataset.py --validate_only

Class YOLO: 0 = signature, 1 = stamp.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CLASS_SIGNATURE = 0
CLASS_STAMP = 1

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

BLUE_INK_COLORS = [
    (5, 30, 85),
    (0, 45, 115),
    (5, 60, 140),
    (20, 80, 165),
    (45, 110, 190),
]
BLACK_INK_COLORS = [
    (8, 12, 18),
    (20, 25, 32),
    (35, 38, 45),
]


@dataclass(frozen=True)
class SignatureAsset:
    name: str
    image: Image.Image
    source_group: str


@dataclass(frozen=True)
class DocumentTemplate:
    name: str
    path: Path
    image: Image.Image
    signature_zones: list[list[float]]
    stamp_zones: list[list[float]]


def iter_image_paths(directory: str | os.PathLike, recursive: bool = True) -> list[Path]:
    root = Path(directory)
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS),
        key=lambda path: str(path).lower(),
    )


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]
    candidates = [Path("C:/Windows/Fonts") / name for name in names]
    candidates.extend([
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ])
    for path in candidates:
        try:
            return ImageFont.truetype(str(path), size)
        except (OSError, ValueError):
            continue
    return ImageFont.load_default()


# -----------------------------------------------------------------------------
# Cau hinh document va vung dat object
# -----------------------------------------------------------------------------

def validate_normalized_zones(zones, source_name: str, field_name: str) -> list[list[float]]:
    if zones is None:
        return []
    if not isinstance(zones, list):
        raise ValueError(f"{source_name}.{field_name} phai la list")

    normalized = []
    for index, zone in enumerate(zones):
        if not isinstance(zone, list) or len(zone) != 4:
            raise ValueError(f"{source_name}.{field_name}[{index}] phai co dang [x1,y1,x2,y2]")
        try:
            x1, y1, x2, y2 = [float(value) for value in zone]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source_name}.{field_name}[{index}] chua gia tri khong phai so") from exc
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(
                f"{source_name}.{field_name}[{index}] phai nam trong 0..1 va co x1<x2, y1<y2"
            )
        normalized.append([x1, y1, x2, y2])
    return normalized


def normalize_region_block(block, source_name: str) -> dict[str, list[list[float]]]:
    if not isinstance(block, dict):
        raise ValueError(f"{source_name} phai la object JSON")
    result = {}
    for field_name in ("signature_zones", "stamp_zones"):
        if field_name in block:
            result[field_name] = validate_normalized_zones(block[field_name], source_name, field_name)
    return result


def load_document_region_config(regions_json: str | os.PathLike) -> dict:
    path = Path(regions_json)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} phai chua mot JSON object")

    documents_raw = raw.get("documents", {})
    if not isinstance(documents_raw, dict):
        raise ValueError("documents trong regions_json phai la object")

    documents = {}
    for document_name, block in documents_raw.items():
        basename = Path(document_name).name
        if basename in documents:
            raise ValueError(f"Trung ten document trong regions_json: {basename}")
        documents[basename] = normalize_region_block(block, f"documents.{basename}")

    return {
        "default": normalize_region_block(raw.get("default", {}), "default"),
        "documents": documents,
    }


def get_regions_for_document(path: str | os.PathLike, region_config: dict | None) -> dict:
    regions = {
        "signature_zones": [zone[:] for zone in DEFAULT_SIGNATURE_ZONES],
        "stamp_zones": [],
    }
    if region_config:
        regions.update(region_config.get("default", {}))
        regions.update(region_config.get("documents", {}).get(Path(path).name, {}))
    return regions


def load_document_backgrounds(
    documents_dir: str | os.PathLike,
    region_config: dict | None = None,
    strict_regions: bool = True,
) -> list[DocumentTemplate]:
    paths = iter_image_paths(documents_dir, recursive=False)
    if not paths:
        return []

    basenames = [path.name for path in paths]
    duplicate_names = sorted({name for name in basenames if basenames.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"Trung basename trong documents_dir: {duplicate_names}")

    if region_config:
        configured = set(region_config.get("documents", {}))
        actual = set(basenames)
        missing = sorted(actual - configured)
        extra = sorted(configured - actual)
        if strict_regions and (missing or extra):
            details = []
            if missing:
                details.append(f"thieu cau hinh: {missing}")
            if extra:
                details.append(f"cau hinh khong co anh: {extra}")
            raise ValueError("regions_json khong khop documents_dir (" + "; ".join(details) + ")")
        if missing:
            print(f"[warn] {len(missing)} document se dung vung mac dinh: {', '.join(missing)}")
        if extra:
            print(f"[warn] {len(extra)} cau hinh khong co document tuong ung: {', '.join(extra)}")

    documents = []
    for path in paths:
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB").copy()
            regions = get_regions_for_document(path, region_config)
            documents.append(DocumentTemplate(
                name=path.name,
                path=path,
                image=image,
                signature_zones=regions.get("signature_zones", []),
                stamp_zones=regions.get("stamp_zones", []),
            ))
        except Exception as exc:
            raise ValueError(f"Khong doc duoc document {path}: {exc}") from exc
    return documents


def normalized_zones_to_pixel_zones(
    zones: Sequence[Sequence[float]],
    content_box: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    cx1, cy1, cx2, cy2 = content_box
    width = cx2 - cx1
    height = cy2 - cy1
    return [
        (
            round(cx1 + width * x1),
            round(cy1 + height * y1),
            round(cx1 + width * x2),
            round(cy1 + height * y2),
        )
        for x1, y1, x2, y2 in zones
    ]


def prepare_document_background(
    document_image: Image.Image,
    width: int,
    height: int,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    source_width, source_height = document_image.size
    if (source_width, source_height) == (width, height):
        return document_image.copy(), (0, 0, width, height)

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = document_image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas.paste(resized, (x, y))
    return canvas, (x, y, x + resized_width, y + resized_height)


# -----------------------------------------------------------------------------
# Tien xu ly bo signatures_dir moi
# -----------------------------------------------------------------------------

def _trim_rgba(image: Image.Image, alpha_threshold: int = 8, padding: int = 2) -> Image.Image | None:
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba)[..., 3]
    ys, xs = np.where(alpha >= alpha_threshold)
    if len(xs) == 0:
        return None
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(rgba.width, int(xs.max()) + padding + 1)
    y2 = min(rgba.height, int(ys.max()) + padding + 1)
    return rgba.crop((x1, y1, x2, y2))


def _estimate_signature_alpha(image: Image.Image) -> np.ndarray:
    rgba = np.asarray(image.convert("RGBA"))
    source_alpha = rgba[..., 3]
    if np.any(source_alpha < 250):
        return source_alpha.copy()

    rgb = rgba[..., :3].astype(np.float32)
    height, width = rgb.shape[:2]
    border_size = max(1, min(height, width) // 20)
    border_pixels = np.concatenate([
        rgb[:border_size].reshape(-1, 3),
        rgb[-border_size:].reshape(-1, 3),
        rgb[:, :border_size].reshape(-1, 3),
        rgb[:, -border_size:].reshape(-1, 3),
    ])
    background = np.median(border_pixels, axis=0)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    background_luma = float(background @ weights)
    luma = rgb @ weights
    color_distance = np.linalg.norm(rgb - background, axis=2)

    # Ket hop do toi va khoang cach mau de tach duoc ca muc den lan muc xanh
    # tren nen trang, xam hoac xanh nhat.
    ink_strength = np.maximum(0.0, background_luma - luma) * 0.75 + color_distance * 0.45
    if float(ink_strength.max()) < 4:
        return np.zeros((height, width), dtype=np.uint8)

    score = np.clip(ink_strength, 0, 255).astype(np.uint8)
    otsu_threshold, _ = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = max(6.0, float(otsu_threshold) * 0.55)
    strong_values = ink_strength[ink_strength > low]
    if strong_values.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    high = max(low + 12.0, float(np.percentile(strong_values, 92)))
    return np.clip((ink_strength - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def _retain_primary_ink_cluster(alpha: np.ndarray) -> np.ndarray:
    """Loai bot chu in roi quanh cac crop ``*_sig_*`` ma van giu cum net ky chinh."""
    height, width = alpha.shape
    binary = (alpha >= 28).astype(np.uint8)
    if int(binary.sum()) < 8:
        return np.zeros_like(alpha)

    kernel_width = max(3, round(width * 0.018))
    kernel_height = max(1, round(height * 0.006))
    joined = cv2.dilate(
        binary,
        np.ones((kernel_height, kernel_width), dtype=np.uint8),
        iterations=1,
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)

    candidates = []
    for component_id in range(1, component_count):
        x, y, box_width, box_height, _ = stats[component_id]
        raw_pixel_count = int(binary[labels == component_id].sum())
        if raw_pixel_count < 4:
            continue
        relative_center_y = (y + box_height / 2) / max(1, height)
        center_weight = 0.75 + 0.25 * (1.0 - abs(relative_center_y - 0.5))
        # Chu ky thuong rong va cao hon dong chu in nho; luy thua chieu cao
        # giup uu tien net viet tay thay vi dong chuc danh ben duoi.
        score = box_width * max(3, box_height) ** 1.55 * center_weight
        candidates.append((score, component_id))

    if not candidates:
        return np.zeros_like(alpha)
    _, primary_id = max(candidates)
    return np.where(labels == primary_id, alpha, 0).astype(np.uint8)


def preprocess_signature(path: Path) -> Image.Image | None:
    with Image.open(path) as opened:
        source = ImageOps.exif_transpose(opened).convert("RGBA")

    alpha = _estimate_signature_alpha(source)
    if "_sig_" in path.name.lower():
        alpha = _retain_primary_ink_cluster(alpha)

    ys, xs = np.where(alpha >= 16)
    if len(xs) < 12:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    content_width = x2 - x1
    content_height = y2 - y1
    if content_width < 8 or content_height < 3:
        return None
    aspect_ratio = content_width / max(1, content_height)
    if not (0.35 <= aspect_ratio <= 22.0):
        return None

    rgba = np.zeros((source.height, source.width, 4), dtype=np.uint8)
    rgba[..., :3] = (8, 45, 125)
    rgba[..., 3] = alpha
    padding = max(1, round(max(source.size) * 0.008))
    return _trim_rgba(Image.fromarray(rgba, mode="RGBA"), alpha_threshold=12, padding=padding)


def load_real_signatures(signatures_dir: str | os.PathLike) -> list[SignatureAsset]:
    paths = iter_image_paths(signatures_dir, recursive=True)
    signatures = []
    rejected = []
    for path in paths:
        try:
            image = preprocess_signature(path)
            if image is None:
                rejected.append(path.name)
                continue
            group = "auto_crop" if "_sig_" in path.name.lower() else "clean_image"
            signatures.append(SignatureAsset(path.name, image, group))
        except Exception as exc:
            rejected.append(path.name)
            print(f"[warn] Bo qua chu ky {path.name}: {exc}")

    if rejected:
        preview = ", ".join(rejected[:5])
        suffix = "..." if len(rejected) > 5 else ""
        print(f"[warn] Loai {len(rejected)} anh chu ky rong/khong hop le: {preview}{suffix}")
    return signatures


def choose_signature_color(blue_probability: float) -> tuple[int, int, int]:
    palette = BLUE_INK_COLORS if random.random() < blue_probability else BLACK_INK_COLORS
    red, green, blue = random.choice(palette)
    jitter = random.randint(-10, 10)
    return tuple(max(0, min(255, channel + jitter)) for channel in (red, green, blue))


def style_signature(asset: SignatureAsset, blue_probability: float) -> Image.Image:
    alpha = np.asarray(asset.image.convert("RGBA"))[..., 3].copy()
    if random.random() < 0.12:
        kernel = np.ones((2, 2), dtype=np.uint8)
        alpha = cv2.dilate(alpha, kernel, iterations=1)
    elif random.random() < 0.10:
        kernel = np.ones((2, 2), dtype=np.uint8)
        alpha = cv2.erode(alpha, kernel, iterations=1)

    gain = random.uniform(0.78, 1.12)
    gamma = random.uniform(0.85, 1.15)
    normalized = np.clip(alpha.astype(np.float32) / 255.0, 0, 1) ** gamma
    alpha = np.clip(normalized * 255.0 * gain, 0, 255).astype(np.uint8)

    color = choose_signature_color(blue_probability)
    rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    rgba[..., :3] = color
    rgba[..., 3] = alpha
    image = Image.fromarray(rgba, mode="RGBA")
    if random.random() < 0.12:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.55)))
    return image


def _bezier_point(p0, p1, p2, p3, t):
    x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
    y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
    return x, y


def generate_fake_signature(width: int = 300, height: int = 120) -> Image.Image:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    color = (*random.choice(BLUE_INK_COLORS), 255)
    cursor_x = width * 0.05
    for _ in range(random.randint(2, 4)):
        stroke_length = random.uniform(width * 0.25, width * 0.5)
        p0 = (cursor_x, random.uniform(height * 0.3, height * 0.7))
        p1 = (cursor_x + stroke_length * 0.3, random.uniform(0, height))
        p2 = (cursor_x + stroke_length * 0.6, random.uniform(0, height))
        p3 = (cursor_x + stroke_length, random.uniform(height * 0.3, height * 0.7))
        points = [_bezier_point(p0, p1, p2, p3, t) for t in np.linspace(0, 1, 60)]
        draw.line(points, fill=color, width=random.randint(2, 4), joint="curve")
        cursor_x += stroke_length * random.uniform(0.7, 0.95)
    if random.random() < 0.5:
        y = height * random.uniform(0.75, 0.9)
        draw.line([(width * 0.05, y), (width * 0.75, y)], fill=color, width=2)
    return _trim_rgba(image.filter(ImageFilter.GaussianBlur(0.35))) or image


# -----------------------------------------------------------------------------
# Con dau procedural
# -----------------------------------------------------------------------------

def generate_stamp(size: int = 180) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    red = random.choice([(160, 0, 0, 225), (180, 8, 8, 215), (145, 0, 15, 220)])
    margin = max(4, round(size * 0.045))
    outer_width = max(2, round(size * 0.022))
    inner_width = max(1, round(size * 0.012))
    draw.ellipse([margin, margin, size - margin, size - margin], outline=red, width=outer_width)
    inset = max(8, round(size * 0.12))
    draw.ellipse([margin + inset, margin + inset, size - margin - inset, size - margin - inset],
                 outline=red, width=inner_width)

    font = load_font(max(8, round(size * 0.075)), bold=True)
    center_text = random.choice(["CTY ABC", "KE TOAN", "DA THU", "PHONG TC-KT"])
    text_box = draw.textbbox((0, 0), center_text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    draw.text(((size - text_width) / 2, (size - text_height) / 2), center_text, fill=red, font=font)

    center = size / 2
    radius = size / 2 - max(12, round(size * 0.14))
    arc_text = "CONG TY TNHH VIET NAM"
    angle_start = 205
    angle_step = 10
    character_size = max(12, round(size * 0.11))
    for index, character in enumerate(arc_text[:16]):
        angle_degrees = angle_start - index * angle_step
        angle = math.radians(angle_degrees)
        x = center + radius * math.cos(angle)
        y = center + radius * math.sin(angle)
        character_image = Image.new("RGBA", (character_size * 2, character_size * 2), (0, 0, 0, 0))
        ImageDraw.Draw(character_image).text((2, 2), character, fill=red, font=font)
        character_image = character_image.rotate(-angle_degrees - 90, expand=True, resample=Image.Resampling.BICUBIC)
        image.paste(character_image, (round(x - character_image.width / 2), round(y - character_image.height / 2)), character_image)

    rgba = np.asarray(image).copy()
    original_alpha = rgba[..., 3]
    noise = np.random.normal(0, 13, original_alpha.shape)
    noisy_alpha = np.clip(original_alpha.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # Khong them noise vao nen trong suot, tranh tao hinh vuong mo quanh con dau.
    rgba[..., 3] = np.where(original_alpha > 0, noisy_alpha, 0)
    result = Image.fromarray(rgba, mode="RGBA")
    result = result.rotate(random.uniform(-10, 10), expand=True, resample=Image.Resampling.BICUBIC)
    return _trim_rgba(result, alpha_threshold=5, padding=2) or result


# -----------------------------------------------------------------------------
# Ghep object vao document
# -----------------------------------------------------------------------------

def fit_asset_to_zone(asset: Image.Image, zone: tuple[int, int, int, int], max_fill: float = 0.96) -> Image.Image:
    x1, y1, x2, y2 = zone
    max_width = max(1, round((x2 - x1) * max_fill))
    max_height = max(1, round((y2 - y1) * max_fill))
    scale = min(1.0, max_width / max(1, asset.width), max_height / max(1, asset.height))
    if scale >= 1:
        return asset
    return asset.resize(
        (max(1, round(asset.width * scale)), max(1, round(asset.height * scale))),
        Image.Resampling.LANCZOS,
    )


def resize_signature_for_zone(
    signature: Image.Image,
    zone: tuple[int, int, int, int],
    signature_scale: float,
) -> Image.Image:
    zone_width = max(1, zone[2] - zone[0])
    width_fill = random.triangular(0.45, 0.96, 0.74) * signature_scale
    target_width = max(8, round(zone_width * width_fill))
    scale = target_width / max(1, signature.width)
    resized = signature.resize(
        (target_width, max(3, round(signature.height * scale))),
        Image.Resampling.LANCZOS,
    )
    resized = resized.rotate(random.uniform(-6, 6), expand=True, resample=Image.Resampling.BICUBIC)
    resized = _trim_rgba(resized, alpha_threshold=5, padding=1) or resized
    return fit_asset_to_zone(resized, zone)


def choose_position_in_zone(
    asset: Image.Image,
    zone: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    center_hint: tuple[float, float] | None = None,
) -> tuple[int, int]:
    x1, y1, x2, y2 = zone
    min_x = max(0, x1)
    min_y = max(0, y1)
    max_x = min(image_width, x2) - asset.width
    max_y = min(image_height, y2) - asset.height
    if center_hint:
        x = round(center_hint[0] - asset.width / 2)
        y = round(center_hint[1] - asset.height / 2)
    else:
        x = random.randint(min_x, max_x) if max_x >= min_x else min_x
        y = random.randint(min_y, max_y) if max_y >= min_y else min_y
    x = max(0, min(x, image_width - asset.width))
    y = max(0, min(y, image_height - asset.height))
    if max_x >= min_x:
        x = max(min_x, min(x, max_x))
    if max_y >= min_y:
        y = max(min_y, min(y, max_y))
    return x, y


def paste_with_alpha(
    background: Image.Image,
    foreground: Image.Image,
    x: int,
    y: int,
    opacity: float = 1.0,
) -> tuple[int, int, int, int]:
    foreground = _trim_rgba(foreground, alpha_threshold=3, padding=0) or foreground.convert("RGBA")
    if opacity < 1:
        alpha = np.asarray(foreground)[..., 3].astype(np.float32)
        alpha = np.clip(alpha * opacity, 0, 255).astype(np.uint8)
        foreground = foreground.copy()
        foreground.putalpha(Image.fromarray(alpha, mode="L"))
    background.paste(foreground, (x, y), foreground)
    return x, y, x + foreground.width, y + foreground.height


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ix1 = max(first[0], second[0])
    iy1 = max(first[1], second[1])
    ix2 = min(first[2], second[2])
    iy2 = min(first[3], second[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return intersection / max(1.0, first_area + second_area - intersection)


def choose_separate_stamp_position(
    stamp: Image.Image,
    stamp_zone: tuple[int, int, int, int],
    signature_boxes: Sequence[Sequence[float]],
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    best_position = choose_position_in_zone(stamp, stamp_zone, image_width, image_height)
    best_score = float("inf")
    zone_diagonal = max(1.0, math.hypot(stamp_zone[2] - stamp_zone[0], stamp_zone[3] - stamp_zone[1]))
    for _ in range(24):
        x, y = choose_position_in_zone(stamp, stamp_zone, image_width, image_height)
        candidate = (x, y, x + stamp.width, y + stamp.height)
        overlap = sum(box_iou(candidate, box) for box in signature_boxes)
        candidate_center = ((candidate[0] + candidate[2]) / 2, (candidate[1] + candidate[3]) / 2)
        nearest_distance = min(
            math.hypot(
                candidate_center[0] - (box[0] + box[2]) / 2,
                candidate_center[1] - (box[1] + box[3]) / 2,
            )
            for box in signature_boxes
        )
        # Uu tien khong de len chu ky, nhung van o gan chu ky co tham quyen.
        score = overlap * 10.0 + nearest_distance / zone_diagonal
        if score < best_score:
            best_score = score
            best_position = (x, y)
    return best_position


def choose_signature_assets(
    signatures_pool: Sequence[SignatureAsset],
    count: int,
    clean_signature_probability: float,
) -> list[SignatureAsset]:
    clean_assets = [asset for asset in signatures_pool if asset.source_group == "clean_image"]
    auto_assets = [asset for asset in signatures_pool if asset.source_group == "auto_crop"]
    random.shuffle(clean_assets)
    random.shuffle(auto_assets)
    selected = []
    used_names = set()

    for _ in range(count):
        prefer_clean = random.random() < clean_signature_probability
        preferred = clean_assets if prefer_clean else auto_assets
        fallback = auto_assets if prefer_clean else clean_assets
        candidate = None
        while preferred and candidate is None:
            option = preferred.pop()
            if option.name not in used_names:
                candidate = option
        while fallback and candidate is None:
            option = fallback.pop()
            if option.name not in used_names:
                candidate = option
        if candidate is None:
            remaining = [asset for asset in signatures_pool if asset.name not in used_names]
            candidate = random.choice(remaining or list(signatures_pool))
        selected.append(candidate)
        used_names.add(candidate.name)
    return selected


def choose_scenario(
    has_signature_zones: bool,
    has_stamp_zones: bool,
    negative_probability: float,
    stamp_probability: float,
    stamp_overlap_probability: float,
) -> str:
    if not has_signature_zones or random.random() < negative_probability:
        return "empty"
    if has_stamp_zones and random.random() < stamp_probability:
        return "sig_and_stamp_overlap" if random.random() < stamp_overlap_probability else "sig_and_stamp_separate"
    return "sig_only"


def compose_sample(
    bg_size: tuple[int, int] = (1000, 1400),
    signatures_pool: Sequence[SignatureAsset] | None = None,
    document: DocumentTemplate | None = None,
    documents_pool: Sequence[DocumentTemplate] | None = None,
    signature_scale: float = 1.0,
    stamp_scale: float = 1.0,
    min_signatures: int = 1,
    max_signatures: int = 5,
    blue_ink_probability: float = 0.78,
    clean_signature_probability: float = 0.70,
    negative_probability: float = 0.08,
    stamp_probability: float = 0.50,
    stamp_overlap_probability: float = 0.55,
) -> tuple[Image.Image, list[tuple[int, int, int, int, int]], dict]:
    width, height = bg_size
    if document is None and documents_pool:
        document = random.choice(list(documents_pool))

    if document:
        image, content_box = prepare_document_background(document.image, width, height)
        signature_zone_defs = document.signature_zones
        stamp_zone_defs = document.stamp_zones
        document_name = document.name
    else:
        image = generate_background(width, height)
        content_box = (0, 0, width, height)
        signature_zone_defs = DEFAULT_SIGNATURE_ZONES
        stamp_zone_defs = []
        document_name = "synthetic"

    signature_zones = normalized_zones_to_pixel_zones(signature_zone_defs, content_box)
    stamp_zones = normalized_zones_to_pixel_zones(stamp_zone_defs, content_box)
    scenario = choose_scenario(
        bool(signature_zones),
        bool(stamp_zones),
        negative_probability,
        stamp_probability,
        stamp_overlap_probability,
    )
    labels: list[tuple[int, int, int, int, int]] = []
    used_signature_names = []
    used_signature_groups = []

    if scenario == "empty":
        return image, labels, {
            "document": document_name,
            "scenario": scenario,
            "signatures": used_signature_names,
            "signature_groups": used_signature_groups,
        }

    minimum = min(max(1, min_signatures), len(signature_zones))
    maximum = min(max(minimum, max_signatures), len(signature_zones))
    signature_count = random.randint(minimum, maximum)
    selected_zones = random.sample(signature_zones, signature_count)

    selected_stamp_zone = random.choice(stamp_zones) if "stamp" in scenario else None
    if selected_stamp_zone:
        stamp_center = (
            (selected_stamp_zone[0] + selected_stamp_zone[2]) / 2,
            (selected_stamp_zone[1] + selected_stamp_zone[3]) / 2,
        )
        authority_zone = min(
            signature_zones,
            key=lambda zone: (
                (zone[0] + zone[2]) / 2 - stamp_center[0]
            ) ** 2 + (
                (zone[1] + zone[3]) / 2 - stamp_center[1]
            ) ** 2,
        )
        if authority_zone not in selected_zones:
            selected_zones[0] = authority_zone

    if signatures_pool:
        selected_assets = choose_signature_assets(
            signatures_pool,
            signature_count,
            clean_signature_probability,
        )
    else:
        selected_assets = [None] * signature_count

    signature_boxes = []
    for zone, asset in zip(selected_zones, selected_assets):
        if asset is None:
            signature = generate_fake_signature(random.randint(220, 340), random.randint(80, 140))
            signature_name = "generated_bezier"
            signature_group = "generated"
        else:
            signature = style_signature(asset, blue_ink_probability)
            signature_name = asset.name
            signature_group = asset.source_group
        signature = resize_signature_for_zone(signature, zone, signature_scale)
        x, y = choose_position_in_zone(signature, zone, width, height)
        box = paste_with_alpha(image, signature, x, y, opacity=random.uniform(0.84, 1.0))
        signature_boxes.append(box)
        labels.append((CLASS_SIGNATURE, *box))
        used_signature_names.append(signature_name)
        used_signature_groups.append(signature_group)

    if selected_stamp_zone:
        zone_width = selected_stamp_zone[2] - selected_stamp_zone[0]
        zone_height = selected_stamp_zone[3] - selected_stamp_zone[1]
        diameter = max(16, round(min(zone_width, zone_height) * random.uniform(0.52, 0.90) * stamp_scale))
        stamp = fit_asset_to_zone(generate_stamp(diameter), selected_stamp_zone, max_fill=0.96)

        if scenario == "sig_and_stamp_overlap":
            stamp_center = (
                (selected_stamp_zone[0] + selected_stamp_zone[2]) / 2,
                (selected_stamp_zone[1] + selected_stamp_zone[3]) / 2,
            )
            target_signature = min(
                signature_boxes,
                key=lambda box: (
                    (box[0] + box[2]) / 2 - stamp_center[0]
                ) ** 2 + (
                    (box[1] + box[3]) / 2 - stamp_center[1]
                ) ** 2,
            )
            center_hint = (
                random.uniform(target_signature[0], target_signature[2]),
                random.uniform(target_signature[1], target_signature[3]),
            )
            stamp_x, stamp_y = choose_position_in_zone(
                stamp, selected_stamp_zone, width, height, center_hint=center_hint
            )
        else:
            stamp_x, stamp_y = choose_separate_stamp_position(
                stamp, selected_stamp_zone, signature_boxes, width, height
            )

        stamp_box = paste_with_alpha(image, stamp, stamp_x, stamp_y, opacity=random.uniform(0.68, 0.92))
        labels.append((CLASS_STAMP, *stamp_box))

    return image, labels, {
        "document": document_name,
        "scenario": scenario,
        "signatures": used_signature_names,
        "signature_groups": used_signature_groups,
    }


# -----------------------------------------------------------------------------
# Nen synthetic fallback
# -----------------------------------------------------------------------------

def generate_background(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = load_font(max(16, round(width * 0.024)), bold=True)
    text_font = load_font(max(11, round(width * 0.014)))
    title = random.choice(["PHIEU CHI", "PHIEU THU", "HOA DON GTGT", "BIEN BAN BAN GIAO"])
    draw.text((round(width * 0.34), round(height * 0.03)), title, fill=(0, 0, 0), font=title_font)
    lines = [
        f"So chung tu: {random.randint(1000, 9999)}",
        f"Ngay lap: {random.randint(1, 28):02d}/{random.randint(1, 12):02d}/2026",
        "Don vi: Cong ty TNHH ABC",
        f"So tien: {random.randint(100_000, 50_000_000):,} VND",
    ]
    y = round(height * 0.08)
    for line in lines:
        draw.text((round(width * 0.06), y), line, fill=(20, 20, 20), font=text_font)
        y += round(height * 0.025)
    table_top = y + round(height * 0.02)
    table_bottom = round(height * 0.65)
    for row in range(7):
        row_y = table_top + round((table_bottom - table_top) * row / 6)
        draw.line([(round(width * 0.06), row_y), (round(width * 0.94), row_y)], fill=(0, 0, 0), width=1)
    for x in (0.06, 0.60, 0.94):
        draw.line([(round(width * x), table_top), (round(width * x), table_bottom)], fill=(0, 0, 0), width=1)
    return image


# -----------------------------------------------------------------------------
# Augmentation va nhan YOLO
# -----------------------------------------------------------------------------

def transform_labels_affine(
    labels: Sequence[Sequence[float]],
    matrix: np.ndarray,
    image_width: int,
    image_height: int,
    min_box_size: int = 4,
) -> list[tuple[int, float, float, float, float]]:
    transformed = []
    matrix = np.asarray(matrix, dtype=np.float32)
    for class_id, x1, y1, x2, y2 in labels:
        corners = np.array([
            [x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1],
        ], dtype=np.float32)
        warped = corners @ matrix.T
        new_x1 = float(np.clip(warped[:, 0].min(), 0, image_width))
        new_y1 = float(np.clip(warped[:, 1].min(), 0, image_height))
        new_x2 = float(np.clip(warped[:, 0].max(), 0, image_width))
        new_y2 = float(np.clip(warped[:, 1].max(), 0, image_height))
        if new_x2 - new_x1 >= min_box_size and new_y2 - new_y1 >= min_box_size:
            transformed.append((class_id, new_x1, new_y1, new_x2, new_y2))
    return transformed


def apply_capture_geometry_augmentation(
    image: Image.Image,
    labels: Sequence[Sequence[float]],
) -> tuple[Image.Image, list[tuple[int, float, float, float, float]]]:
    if random.random() > 0.72:
        return image, list(labels)
    array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2),
        random.uniform(-2.2, 2.2),
        random.uniform(0.97, 1.025),
    )
    matrix[0, 2] += random.uniform(-0.018, 0.018) * width
    matrix[1, 2] += random.uniform(-0.018, 0.018) * height
    warped = cv2.warpAffine(
        array,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return Image.fromarray(warped), transform_labels_affine(labels, matrix, width, height)


def add_uneven_lighting(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    xs = np.linspace(-1, 1, width, dtype=np.float32)
    ys = np.linspace(-1, 1, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    angle = random.uniform(0, math.pi)
    direction = math.cos(angle) * grid_x + math.sin(angle) * grid_y
    direction = (direction - direction.min()) / max(1e-6, direction.max() - direction.min())
    gradient = 1.0 + (direction - 0.5) * random.uniform(0.08, 0.22)
    return np.clip(array.astype(np.float32) * gradient[..., None], 0, 255).astype(np.uint8)


def apply_scan_augmentation(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("RGB"))
    if random.random() < 0.55:
        noise = np.random.normal(0, random.uniform(2, 8), array.shape).astype(np.int16)
        array = np.clip(array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.30:
        array = cv2.GaussianBlur(array, (random.choice([3, 3, 5]),) * 2, 0)
    if random.random() < 0.28:
        array = cv2.convertScaleAbs(array, alpha=random.uniform(0.88, 1.04), beta=random.uniform(-8, 8))
    if random.random() < 0.30:
        array = add_uneven_lighting(array)
    output = Image.fromarray(array)
    if random.random() < 0.42:
        buffer = io.BytesIO()
        output.save(buffer, format="JPEG", quality=random.randint(52, 84))
        buffer.seek(0)
        output = Image.open(buffer).convert("RGB")
    return output


def save_yolo_label(
    labels: Sequence[Sequence[float]],
    image_width: int,
    image_height: int,
    label_path: str | os.PathLike,
) -> None:
    lines = []
    for class_id, x1, y1, x2, y2 in labels:
        x1 = max(0.0, min(float(image_width), float(x1)))
        x2 = max(0.0, min(float(image_width), float(x2)))
        y1 = max(0.0, min(float(image_height), float(y1)))
        y2 = max(0.0, min(float(image_height), float(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        center_x = (x1 + x2) / 2 / image_width
        center_y = (y1 + y2) / 2 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        lines.append(f"{int(class_id)} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    Path(label_path).write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI va lich sinh can bang document
# -----------------------------------------------------------------------------

def parse_image_sizes(raw: str) -> list[tuple[int, int]]:
    sizes = []
    for item in raw.split(","):
        item = item.strip().lower().replace(" ", "")
        if not item:
            continue
        try:
            width_text, height_text = item.split("x", 1)
            width, height = int(width_text), int(height_text)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Kich thuoc khong dung dinh dang WxH: {item}") from exc
        if width <= 0 or height <= 0:
            raise ValueError(f"Kich thuoc phai > 0: {item}")
        sizes.append((width, height))
    if not sizes:
        raise ValueError("--img_sizes khong co kich thuoc hop le")
    return sizes


def choose_output_size(
    document: DocumentTemplate | None,
    size_profile: str,
    fixed_size: tuple[int, int],
    custom_sizes: Sequence[tuple[int, int]] | None,
) -> tuple[int, int]:
    if custom_sizes:
        candidates = list(custom_sizes)
    elif size_profile == "native" and document:
        return document.image.size
    elif size_profile == "fixed":
        return fixed_size
    else:
        candidates = list(DEFAULT_REALISTIC_SIZES)

    if document:
        is_landscape = document.image.width > document.image.height
        matching = [size for size in candidates if (size[0] > size[1]) == is_landscape]
        if matching:
            candidates = matching
    return random.choice(candidates)


def balanced_document_schedule(
    documents: Sequence[DocumentTemplate],
    sample_count: int,
) -> list[DocumentTemplate | None]:
    if not documents:
        return [None] * sample_count
    schedule = []
    while len(schedule) < sample_count:
        cycle = list(documents)
        random.shuffle(cycle)
        schedule.extend(cycle)
    return schedule[:sample_count]


def validate_probability(value: float, argument_name: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{argument_name} phai nam trong khoang 0..1")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sinh dataset YOLO tu signatures_dir va documents")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--out_dir", default="dataset_generated")
    parser.add_argument("--signatures_dir", default="signatures_dir")
    parser.add_argument("--documents_dir", default="documents")
    parser.add_argument("--regions_json", default="document_regions.json")
    parser.add_argument("--signature_scale", type=float, default=1.0)
    parser.add_argument("--stamp_scale", type=float, default=1.0)
    parser.add_argument("--min_signatures", type=int, default=1)
    parser.add_argument("--max_signatures", type=int, default=5)
    parser.add_argument("--blue_ink_probability", type=float, default=0.78)
    parser.add_argument("--clean_signature_probability", type=float, default=0.70,
                        help="Xac suat uu tien nhom image/Screenshot sach thay vi crop *_sig_* co the dinh chu in")
    parser.add_argument("--negative_probability", type=float, default=0.08)
    parser.add_argument("--stamp_probability", type=float, default=0.50)
    parser.add_argument("--stamp_overlap_probability", type=float, default=0.55)
    parser.add_argument("--size_profile", choices=["native", "fixed", "realistic"], default="native")
    parser.add_argument("--img_w", type=int, default=1191)
    parser.add_argument("--img_h", type=int, default=1685)
    parser.add_argument("--img_sizes", default=None,
                        help="Danh sach kich thuoc WxH cach nhau boi dau phay; uu tien dung kich thuoc cung chieu document")
    parser.add_argument("--no_geometry_aug", action="store_true")
    parser.add_argument("--no_scan_aug", action="store_true")
    parser.add_argument("--allow_fake_signatures", action="store_true",
                        help="Cho phep fallback chu ky Bezier khi signatures_dir khong hop le/rong")
    parser.add_argument("--allow_synthetic_documents", action="store_true",
                        help="Cho phep fallback nen tu ve khi documents_dir khong hop le/rong")
    parser.add_argument("--allow_region_mismatch", action="store_true",
                        help="Khong dung chuong trinh khi regions_json thieu/thua ten document")
    parser.add_argument("--overwrite", action="store_true",
                        help="Cho phep ghi de cac sample trung ten trong out_dir; khong xoa file cu")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.num_samples <= 0:
        raise ValueError("--num_samples phai > 0")
    if args.signature_scale <= 0 or args.stamp_scale <= 0:
        raise ValueError("--signature_scale va --stamp_scale phai > 0")
    if args.min_signatures <= 0 or args.max_signatures < args.min_signatures:
        raise ValueError("Can 0 < --min_signatures <= --max_signatures")
    if args.img_w <= 0 or args.img_h <= 0:
        raise ValueError("--img_w va --img_h phai > 0")
    validate_probability(args.blue_ink_probability, "--blue_ink_probability")
    validate_probability(args.clean_signature_probability, "--clean_signature_probability")
    validate_probability(args.negative_probability, "--negative_probability")
    validate_probability(args.stamp_probability, "--stamp_probability")
    validate_probability(args.stamp_overlap_probability, "--stamp_overlap_probability")

    signatures_path = Path(args.signatures_dir)
    if signatures_path.is_dir():
        signatures = load_real_signatures(signatures_path)
    else:
        signatures = []
    if not signatures and not args.allow_fake_signatures:
        raise FileNotFoundError(
            f"Khong load duoc chu ky nao tu {signatures_path}. "
            "Dung --allow_fake_signatures neu muon fallback Bezier."
        )

    regions_path = Path(args.regions_json)
    region_config = load_document_region_config(regions_path) if regions_path.is_file() else None
    if not region_config and not args.allow_region_mismatch:
        raise FileNotFoundError(f"Khong tim thay regions_json: {regions_path}")

    documents_path = Path(args.documents_dir)
    documents = load_document_backgrounds(
        documents_path,
        region_config=region_config,
        strict_regions=not args.allow_region_mismatch,
    ) if documents_path.is_dir() else []
    if not documents and not args.allow_synthetic_documents:
        raise FileNotFoundError(
            f"Khong load duoc document nao tu {documents_path}. "
            "Dung --allow_synthetic_documents neu muon fallback nen tu ve."
        )

    group_counts = {
        group: sum(asset.source_group == group for asset in signatures)
        for group in sorted({asset.source_group for asset in signatures})
    }
    print(f"Da load {len(signatures)} chu ky hop le tu {signatures_path}")
    if group_counts:
        print("  " + ", ".join(f"{name}: {count}" for name, count in group_counts.items()))
    print(f"Da load {len(documents)} document tu {documents_path}")
    if documents:
        portrait = sum(document.image.height > document.image.width for document in documents)
        landscape = len(documents) - portrait
        signature_zone_count = sum(len(document.signature_zones) for document in documents)
        stamp_zone_count = sum(len(document.stamp_zones) for document in documents)
        print(
            f"  portrait: {portrait}, landscape: {landscape}, "
            f"signature zones: {signature_zone_count}, stamp zones: {stamp_zone_count}"
        )

    if args.validate_only:
        print("Kiem tra dau vao thanh cong; khong sinh dataset (--validate_only).")
        return

    custom_sizes = parse_image_sizes(args.img_sizes) if args.img_sizes else None
    output_root = Path(args.out_dir)
    image_dir = output_root / "images"
    label_dir = output_root / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    target_names = [f"sample_{index:05d}" for index in range(args.num_samples)]
    collisions = [
        name for name in target_names
        if (image_dir / f"{name}.jpg").exists() or (label_dir / f"{name}.txt").exists()
    ]
    if collisions and not args.overwrite:
        raise FileExistsError(
            f"{len(collisions)} sample da ton tai trong {output_root}. "
            "Chon out_dir moi hoac dung --overwrite de ghi de dung cac ten trung."
        )

    schedule = balanced_document_schedule(documents, args.num_samples)
    manifest_path = output_root / "manifest.jsonl"
    manifest_mode = "w" if args.overwrite or not manifest_path.exists() else "a"
    signature_label_count = 0
    stamp_label_count = 0
    scenario_counts: dict[str, int] = {}

    with manifest_path.open(manifest_mode, encoding="utf-8") as manifest:
        for index, document in enumerate(schedule):
            output_size = choose_output_size(
                document,
                args.size_profile,
                (args.img_w, args.img_h),
                custom_sizes,
            )
            image, labels, metadata = compose_sample(
                bg_size=output_size,
                signatures_pool=signatures,
                document=document,
                signature_scale=args.signature_scale,
                stamp_scale=args.stamp_scale,
                min_signatures=args.min_signatures,
                max_signatures=args.max_signatures,
                blue_ink_probability=args.blue_ink_probability,
                clean_signature_probability=args.clean_signature_probability,
                negative_probability=args.negative_probability,
                stamp_probability=args.stamp_probability,
                stamp_overlap_probability=args.stamp_overlap_probability,
            )
            if not args.no_geometry_aug:
                image, labels = apply_capture_geometry_augmentation(image, labels)
            if not args.no_scan_aug:
                image = apply_scan_augmentation(image)

            sample_name = target_names[index]
            image.save(image_dir / f"{sample_name}.jpg", quality=92, subsampling=0)
            save_yolo_label(labels, image.width, image.height, label_dir / f"{sample_name}.txt")

            signature_label_count += sum(label[0] == CLASS_SIGNATURE for label in labels)
            stamp_label_count += sum(label[0] == CLASS_STAMP for label in labels)
            scenario = metadata["scenario"]
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
            metadata.update({
                "sample": sample_name,
                "width": image.width,
                "height": image.height,
                "label_count": len(labels),
            })
            manifest.write(json.dumps(metadata, ensure_ascii=False) + "\n")

            if (index + 1) % 100 == 0 or index + 1 == args.num_samples:
                print(f"Da sinh {index + 1}/{args.num_samples} samples")

    yaml_path = output_root / "data_unsplit_DO_NOT_TRAIN.yaml"
    yaml_path.write_text(
        "# Dataset chua split; hay chay split_dataset.py truoc khi train.\n"
        f"path: {output_root.resolve().as_posix()}\n"
        "train: images\n"
        "val: images\n"
        "names:\n"
        "  0: signature\n"
        "  1: stamp\n",
        encoding="utf-8",
    )

    print(f"Hoan tat: {output_root.resolve()}")
    print(f"Nhan signature: {signature_label_count}; stamp: {stamp_label_count}")
    print("Scenario: " + ", ".join(f"{name}={count}" for name, count in sorted(scenario_counts.items())))
    print(f"Manifest: {manifest_path}")
    print(
        f"Buoc tiep theo: python split_dataset.py --src_dir {args.out_dir} "
        f"--dst_dir {args.out_dir}_split --val_ratio 0.15"
    )


if __name__ == "__main__":
    main()
