"""
split_dataset.py
==================
Chia dataset sinh ra từ generate_dataset.py (thư mục images/ + labels/ chung)
thành cấu trúc chuẩn YOLO: train/ và val/ riêng biệt, kèm data.yaml tương ứng.

CÁCH DÙNG:
    python split_dataset.py --src_dir dataset --dst_dir dataset_split --val_ratio 0.15
"""

import os
import shutil
import random
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True,
                            help="Thư mục dataset gốc (có images/ và labels/ chung)")
    parser.add_argument("--dst_dir", type=str, required=True,
                            help="Thư mục đích, sẽ tạo train/ val/ bên trong")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    img_dir = os.path.join(args.src_dir, "images")
    lbl_dir = os.path.join(args.src_dir, "labels")

    files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(files)

    n_val = max(1, int(len(files) * args.val_ratio))
    val_files = set(files[:n_val])
    train_files = files[n_val:]

    for split_name, split_files in [("train", train_files), ("val", val_files)]:
        os.makedirs(os.path.join(args.dst_dir, split_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.dst_dir, split_name, "labels"), exist_ok=True)
        for fname in split_files:
            base = os.path.splitext(fname)[0]
            shutil.copy(os.path.join(img_dir, fname),
                        os.path.join(args.dst_dir, split_name, "images", fname))
            lbl_src = os.path.join(lbl_dir, base + ".txt")
            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, os.path.join(args.dst_dir, split_name, "labels", base + ".txt"))

    yaml_path = os.path.join(args.dst_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {os.path.abspath(args.dst_dir)}\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"names:\n  0: signature\n  1: stamp\n"
        )

    print(f"Train: {len(train_files)} anh | Val: {len(val_files)} anh")
    print(f"Da tao data.yaml tai: {yaml_path}")


if __name__ == "__main__":
    main()
