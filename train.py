"""
train.py
=========
Train model YOLO11 phát hiện signature/stamp trên ảnh chứng từ.

Các lựa chọn cấu hình trong script này được chọn RIÊNG cho đặc thù bài toán:
  - imgsz=1024 (thay vi mac dinh 640): vi signature/stamp la object nho so voi
    ca trang tai lieu, giam resolution se lam mat chi tiet net chu ky mong.
  - model mac dinh yolo11s (khong phai n) de co du capacity phan biet net chu ky
    mong voi nen/duong ke bang; neu can nhe hon cho production co the đổi 'n'.
  - patience cao hon mac dinh vi dataset synthetic hoi "de" luc dau, tranh early
    stop qua som truoc khi model on dinh tren cac case kho (overlap).
  - augmentation hinh hoc (mosaic, flip) duoc GIAM bot so voi mac dinh vi:
      + KHONG nen flip ngang/doc: chu ky/dau bi lat se khong con giong chu ky/dau that
      + mosaic vua phai vi de sinh nhieu context nen khac nhau, nhung qua nhieu
        se pha vo vi tri "hop ly" cua chu ky (thuong o cuoi trang)

CACH DUNG:
    python train.py --data dataset_split/data.yaml --epochs 100 --imgsz 1024
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Duong dan data.yaml")
    parser.add_argument("--model", type=str, default="yolo11s.pt",
                          help="Pretrained weight khoi tao (transfer learning). "
                              "Dung yolo11n.pt neu can nhe/nhanh hon cho production.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--project", type=str, default="runs_signature")
    parser.add_argument("--name", type=str, default="yolo11_sig_stamp")
    parser.add_argument("--device", type=str, default="0",
                          help="'0' = GPU dau tien, '0,1' = 2 GPU, 'cpu' = ep dung CPU. "
                              "Neu khong truyen, Ultralytics tu chon GPU neu co san.")
    parser.add_argument("--workers", type=int, default=4,
                          help="So luong dataloader worker. Tang len khi dung GPU de "
                              "tranh GPU bi 'doi' du lieu (bottleneck o CPU/disk).")
    args = parser.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
        # --- Augmentation: tat cac phep bien doi khong hop ly voi chu ky/dau ---
        fliplr=0.0,      # KHONG lat ngang: chu ky bi lat khong con giong that
        flipud=0.0,      # KHONG lat doc
        mosaic=0.3,      # giam mosaic (mac dinh 1.0) de giu boi canh layout tai lieu
        mixup=0.0,       # tat mixup: tron 2 anh lam mo net chu ky, phan tac dung

        # --- Cac phep bien doi con lai giu muc vua phai, mo phong bien dang scan ---
        degrees=3.0,     # xoay nhe, giong nghieng trang khi scan
        translate=0.1,
        scale=0.3,
        shear=0.0,
        hsv_h=0.01, hsv_s=0.3, hsv_v=0.3,  # bien doi mau/anh sang vua phai

        # --- Ho tro object nho ---
        close_mosaic=10,  # tat mosaic o 10 epoch cuoi de model "quen" voi ty le that
    )

    # Chay validate cuoi cung, in ra metric tong quat
    metrics = model.val(data=args.data, imgsz=args.imgsz)
    print("\n=== KET QUA VALIDATION TONG QUAT ===")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print("\nChay them 'python eval_per_class.py' de xem chi tiet Precision/Recall "
          "TUNG LOP (signature vs stamp) -- quan trong de kiem tra xem signature "
          "co bi 'lep ve' so voi stamp hay khong.")


if __name__ == "__main__":
    main()
