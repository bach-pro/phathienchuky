"""Quet anh chup tai lieu: cat trang, sua phoi canh va lam sach nen.

Vi du:
    python scan.py --input test/imagegg.png --output test/scanned_output.png
    python scan.py --input test/imagegg.png --mode bw --preview

Mac dinh luu anh mau de giu nguyen mau con dau va chu ky. Dung ``--mode gray``
hoac ``--mode bw`` khi can anh xam/trang-den cho OCR.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def order_points(points: np.ndarray) -> np.ndarray:
    """Sap xep bon diem theo TL, TR, BR, BL."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.empty((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def resize_for_detection(image: np.ndarray, max_side: int = 1600) -> tuple[np.ndarray, float]:
    """Thu nho anh de tim bien nhanh, tra ve anh va ti le so voi anh goc."""
    height, width = image.shape[:2]
    scale = min(1.0, max_side / float(max(height, width)))
    if scale == 1.0:
        return image.copy(), scale
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def find_document_corners(image: np.ndarray) -> np.ndarray | None:
    """Tim tu giac co kha nang la trang giay nhat trong anh."""
    preview, scale = resize_for_detection(image)
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Tu dong dat nguong Canny theo trung vi anh, sau do noi cac canh bi dut.
    median = float(np.median(gray))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, max(lower + 30, 1.33 * median)))
    edges = cv2.Canny(gray, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = preview.shape[0] * preview.shape[1]
    candidates: list[tuple[float, np.ndarray]] = []

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.15:
            continue
        perimeter = cv2.arcLength(contour, True)
        for epsilon in (0.015, 0.02, 0.025, 0.03, 0.04):
            polygon = cv2.approxPolyDP(contour, epsilon * perimeter, True)
            if len(polygon) == 4 and cv2.isContourConvex(polygon):
                # Uu tien tu giac lon; hinh chu nhat qua mong khong phai trang giay.
                rect = order_points(polygon.reshape(4, 2))
                top = np.linalg.norm(rect[1] - rect[0])
                bottom = np.linalg.norm(rect[2] - rect[3])
                left = np.linalg.norm(rect[3] - rect[0])
                right = np.linalg.norm(rect[2] - rect[1])
                width = max(top, bottom)
                height = max(left, right)
                if min(width, height) < 0.20 * max(width, height):
                    break
                candidates.append((area, rect / scale))
                break

    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def warp_document(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Hieu chinh phoi canh tu bon goc tai lieu."""
    tl, tr, br, bl = order_points(corners)
    width = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    if width < 2 or height < 2:
        raise ValueError("Bon goc tai lieu khong hop le.")

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(order_points(corners), destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Khử bong va nen khong deu ma van giu net muc manh."""
    short_side = min(gray.shape[:2])
    sigma = max(15.0, short_side / 40.0)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = cv2.divide(gray, np.maximum(background, 1), scale=245)
    output = np.empty_like(normalized, dtype=np.uint8)
    cv2.normalize(
        normalized,
        output,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    return output


def normalize_color_background(image: np.ndarray) -> np.ndarray:
    """Lam trang nen giay theo tung kenh ma khong khu mau muc/dau."""
    short_side = min(image.shape[:2])
    sigma = max(15.0, short_side / 40.0)
    source = image.astype(np.float32)
    background = cv2.GaussianBlur(source, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # Chia cho nen cuc bo de loai bong, anh sang vang va vung xam khong deu.
    flattened = cv2.divide(source, np.maximum(background, 1.0), scale=245.0)
    flattened = np.clip(flattened, 0, 255).astype(np.uint8)

    # Chi dua cac pixel sang va gan trung tinh ve trang. Pixel co mau bao hoa
    # (dau do, chu ky xanh) va chu toi se khong nam trong mask nay.
    darkest = flattened.min(axis=2).astype(np.float32)
    brightest = flattened.max(axis=2).astype(np.float32)
    chroma = brightest - darkest
    luminance = flattened.mean(axis=2)
    neutral_bright = (chroma < 22.0) & (luminance > 165.0)
    strength = np.clip((luminance - 165.0) / 55.0, 0.0, 1.0)
    strength = np.where(neutral_bright, strength, 0.0)[..., None]
    whitened = flattened.astype(np.float32) * (1.0 - strength) + 255.0 * strength
    return np.clip(whitened, 0, 255).astype(np.uint8)


def enhance_scan(image: np.ndarray, mode: str = "color") -> np.ndarray:
    """Lam sach anh theo che do color, gray hoac bw."""
    if mode == "color":
        result = normalize_color_background(image)
        # Loc nhe nhieu mau camera, khong lam nhoe net muc mong.
        return cv2.bilateralFilter(result, 5, 20, 20)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = normalize_illumination(gray)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if mode == "gray":
        return cv2.GaussianBlur(gray, (3, 3), 0)

    # Threshold khoi lon, le de OpenCV chap nhan va tu co gian theo do phan giai.
    block_size = max(31, (min(gray.shape[:2]) // 20) | 1)
    block_size = min(block_size, 151)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        12,
    )


def scan_document(image: np.ndarray, mode: str = "color") -> tuple[np.ndarray, bool]:
    """Tra ve anh scan va trang thai co tim thay bien trang hay khong."""
    corners = find_document_corners(image)
    found_page = corners is not None
    corrected = warp_document(image, corners) if found_page else image.copy()
    return enhance_scan(corrected, mode), found_page


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Scan va hieu chinh anh chup tai lieu.")
    parser.add_argument("--input", type=Path, default=base_dir / "test" / "imagegg.png")
    parser.add_argument("--output", type=Path, default=base_dir / "test" / "scanned_output.png")
    parser.add_argument(
        "--mode",
        choices=("color", "gray", "bw"),
        default="color",
        help="color giu mau dau/chu ky; gray hoac bw phu hop OCR.",
    )
    parser.add_argument("--preview", action="store_true", help="Mo cua so xem ket qua.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {args.input}")

    scanned, found_page = scan_document(image, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), scanned):
        raise OSError(f"Khong ghi duoc anh: {args.output}")

    status = "da cat va sua phoi canh" if found_page else "khong tim thay bien trang, giu khung anh goc"
    print(f"Da luu anh scan ({args.mode}, {status}): {args.output}")

    if args.preview:
        cv2.imshow("Document scan", scanned)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
