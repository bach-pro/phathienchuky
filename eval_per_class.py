"""
eval_per_class.py
===================
Đánh giá model đã train, in ra Precision/Recall/mAP RIÊNG cho từng lớp
(signature vs stamp). Đây là bước quan trọng vì:

  - Metric tổng (mAP chung) có thể "che giấu" việc model học tốt stamp
    (dễ học vì hình dạng đều, màu đỏ nổi bật) nhưng học kém signature
    (khó học vì hình dạng bất định, dễ bị occlude bởi stamp).
  - Với bài toán nghiệp vụ kiểm tra chứng từ, RECALL của signature quan trọng
    hơn Precision (bỏ sót chữ ký nguy hiểm hơn báo nhầm) -- script này cảnh báo
    riêng nếu recall signature thấp hơn ngưỡng chấp nhận được.

CACH DUNG:
    uv run eval_per_class.py --weights runs\\detect\\runs_signature\\yolov8_sig_stamp-9\\weights\\best.pt --data dataset_split/data.yaml
"""

import argparse
from ultralytics import YOLO

CLASS_NAMES = {0: "signature", 1: "stamp"}
MIN_ACCEPTABLE_RECALL_SIGNATURE = 0.85  # nguong nghiep vu de canh bao


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Duong dan best.pt")
    parser.add_argument("--data", type=str, required=True, help="Duong dan data.yaml")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25,
                            help="Nguong confidence de tinh Precision/Recall diem nay. "
                                "Neu uu tien recall cao (khuyen nghi cho bai toan nay), "
                                "co the giam xuong 0.1-0.15.")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, conf=args.conf)

    print("\n" + "=" * 60)
    print("KET QUA CHI TIET THEO TUNG LOP")
    print("=" * 60)

    # metrics.box.p, .r, .ap50, .ap la array theo thu tu class index
    p_per_class = metrics.box.p
    r_per_class = metrics.box.r
    ap50_per_class = metrics.box.ap50
    ap_per_class = metrics.box.ap

    warnings = []

    for idx, name in CLASS_NAMES.items():
        try:
            precision = p_per_class[idx]
            recall = r_per_class[idx]
            ap50 = ap50_per_class[idx]
            ap = ap_per_class[idx]
        except (IndexError, TypeError):
            print(f"[{name}] Khong co du lieu (co the khong co sample nao trong val set)")
            continue

        print(f"\nLop: {name}")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  mAP50     : {ap50:.4f}")
        print(f"  mAP50-95  : {ap:.4f}")

        if name == "signature" and recall < MIN_ACCEPTABLE_RECALL_SIGNATURE:
            warnings.append(
                f"[CANH BAO] Recall cua 'signature' = {recall:.4f}, "
                f"THAP hon nguong nghiep vu ({MIN_ACCEPTABLE_RECALL_SIGNATURE}). "
                f"Model dang co nguy co BO SOT chu ky -- rui ro cao cho nghiep vu "
                f"kiem tra chung tu. Goi y khac phuc:\n"
                f"    1. Giam --conf khi inference (vd 0.1-0.15) de tang recall\n"
                f"    2. Them nhieu sample 'signature bi stamp che' vao training set\n"
                f"    3. Tang trong so loss cho class signature (cls weight)\n"
                f"    4. Kiem tra lai anchor/resolution -- co the object qua nho"
            )

    # So sanh cheo: neu stamp recall vuot troi han signature -> dau hieu mat can bang
    try:
        r_sig = r_per_class[0]
        r_stamp = r_per_class[1]
        if r_stamp - r_sig > 0.15:
            warnings.append(
                f"[CANH BAO] Chenh lech recall giua stamp ({r_stamp:.4f}) va "
                f"signature ({r_sig:.4f}) qua lon (>0.15). Day la dau hieu dien hinh "
                f"cua viec model 'thien vi' hoc dac trung de (dau mau do, hinh tron deu) "
                f"va bo qua dac trung kho hon (net chu ky mong, bat dinh hinh) -- dung "
                f"nhu van de occlusion da trao doi truoc do."
            )
    except (IndexError, TypeError):
        pass

    if warnings:
        print("\n" + "=" * 60)
        print("CAC CANH BAO NGHIEP VU")
        print("=" * 60)
        for w in warnings:
            print("\n" + w)
    else:
        print("\nKhong co canh bao -- Recall cua signature dat nguong chap nhan duoc.")


if __name__ == "__main__":
    main()
