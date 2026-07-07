"""
train_sweep.py
==============
Chay lan luot nhieu cau hinh train de so sanh tren cung dataset.

Vi du baseline theo lo trinh:
    python train_sweep.py --data dataset_documents_multisig_split/data.yaml --epochs 100 --batch 4

Xem truoc lenh, chua train:
    python train_sweep.py --data dataset_documents_multisig_split/data.yaml --dry_run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_csv(raw):
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Danh sach khong duoc rong")
    return values


def parse_imgszs(raw):
    imgszs = []
    for item in parse_csv(raw):
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"imgsz khong hop le: {item}") from exc
        if value <= 0:
            raise ValueError(f"imgsz phai > 0: {item}")
        imgszs.append(value)
    return imgszs


def build_run_name(model_path, imgsz, prefix):
    stem = Path(model_path).stem
    return f"{prefix}_{stem}_imgsz{imgsz}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Duong dan data.yaml da split.")
    parser.add_argument("--models", default="yolo11n.pt,yolo11s.pt,yolo11l.pt",
                        help="Danh sach weight cach nhau bang dau phay.")
    parser.add_argument("--imgszs", default="1024,1280",
                        help="Danh sach imgsz cach nhau bang dau phay.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--project", default="runs_signature")
    parser.add_argument("--name_prefix", default="sweep")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only_missing", action="store_true",
                        help="Bo qua run da co weights/best.pt.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Chi in cac lenh train, khong chay.")
    args = parser.parse_args()

    models = parse_csv(args.models)
    imgszs = parse_imgszs(args.imgszs)

    if args.epochs <= 0:
        raise ValueError("--epochs phai > 0")
    if args.batch == 0:
        raise ValueError("--batch khong duoc = 0")

    commands = []
    for model_path in models:
        for imgsz in imgszs:
            run_name = build_run_name(model_path, imgsz, args.name_prefix)
            best_path = Path(args.project) / run_name / "weights" / "best.pt"
            if args.only_missing and best_path.exists():
                print(f"[skip] Da co {best_path}")
                continue

            commands.append([
                sys.executable,
                "train.py",
                "--data", args.data,
                "--model", model_path,
                "--epochs", str(args.epochs),
                "--imgsz", str(imgsz),
                "--batch", str(args.batch),
                "--patience", str(args.patience),
                "--project", args.project,
                "--name", run_name,
                "--device", args.device,
                "--workers", str(args.workers),
            ])

    if not commands:
        print("Khong co cau hinh nao can chay.")
        return

    print("Cac cau hinh se chay:")
    for command in commands:
        print("  " + " ".join(command))

    if args.dry_run:
        return

    for idx, command in enumerate(commands, start=1):
        print("\n" + "=" * 80)
        print(f"RUN {idx}/{len(commands)}: {' '.join(command)}")
        print("=" * 80)
        env = os.environ.copy()
        completed = subprocess.run(command, env=env)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
