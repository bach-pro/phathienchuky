# Phat hien chu ky va con dau bang YOLO

Project nay dung YOLO/Ultralytics de phat hien 2 lop tren anh chung tu:

- `signature`: chu ky
- `stamp`: con dau/moc

Code ho tro sinh dataset synthetic, chia train/val, train model, danh gia theo tung lop, phan tich loi va predict tren mot anh bat ky.

## 1. Yeu cau

- Python 3.10 tro len
- Nen dung GPU NVIDIA neu train voi anh lon, nhung van co the chay CPU cho test nho
- Windows PowerShell hoac terminal tuong duong

## 2. Cai dat

Tao moi truong ao:

```powershell
python -m venv env
.\env\Scripts\activate
```

Cap nhat `pip` va cai thu vien:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kiem tra nhanh:

```powershell
python -c "from ultralytics import YOLO; import cv2; print('OK')"
```

Neu muon train bang GPU, hay dam bao PyTorch nhan CUDA dung voi may cua ban. Co the kiem tra bang:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

## 3. Cau truc thu muc quan trong

- `generate_dataset.py`: sinh dataset synthetic gom anh chung tu, chu ky va con dau
- `split_dataset.py`: chia dataset thanh `train` va `val`, tao `data.yaml`
- `train.py`: train YOLO cho 2 lop `signature` va `stamp`
- `train_sweep.py`: chay nhieu cau hinh train de so sanh model/imgsz
- `eval_per_class.py`: in Precision/Recall/mAP rieng cho tung lop
- `analyze_errors.py`: xuat CSV false positive/false negative de phan tich loi
- `predict_single_image.py`: predict tren mot anh, ve box, cat crop va luu JSON toa do
- `documents/`: anh mau chung tu neu co
- `signatures_dir/`: anh chu ky that neu co
- `test/`: anh dung de thu predict
- `runs_signature/` hoac `runs/`: output train cua Ultralytics

## 4. Sinh dataset synthetic

Lenh co ban:

```powershell
python generate_dataset.py --num_samples 2000 --out_dir dataset_documents_multisig
```

Neu co anh chung tu mau va chu ky that:

```powershell
python generate_dataset.py `
  --num_samples 2000 `
  --out_dir dataset_documents_multisig `
  --documents_dir documents `
  --signatures_dir signatures_dir `
  --regions_json document_regions.json `
  --signature_scale 1.875 `
  --stamp_scale 2.0 `
  --min_signatures 2 `
  --max_signatures 5
```

Output se nam trong:

```text
dataset_documents_multisig/
  images/
  labels/
  data_unsplit_DO_NOT_TRAIN.yaml
```

Luu y: khong dung file `data_unsplit_DO_NOT_TRAIN.yaml` de train, vi train va val se tro chung mot tap anh.

## 5. Chia train/val

```powershell
python split_dataset.py --src_dir dataset_documents_multisig --dst_dir dataset_documents_multisig_split --val_ratio 0.15
```

Sau buoc nay se co:

```text
dataset_documents_multisig_split/
  train/
    images/
    labels/
  val/
    images/
    labels/
  data.yaml
```

## 6. Train model

Train mot cau hinh:

```powershell
python train.py `
  --data dataset_documents_multisig_split/data.yaml `
  --model yolo11s.pt `
  --epochs 100 `
  --imgsz 1024 `
  --batch 4 `
  --device 0
```

Neu khong co GPU, dung CPU:

```powershell
python train.py --data dataset_documents_multisig_split/data.yaml --model yolo11s.pt --epochs 50 --imgsz 1024 --batch 2 --device cpu
```

Ket qua train thuong nam tai:

```text
runs_signature/yolo11_sig_stamp/weights/best.pt
```

## 7. Chay sweep nhieu cau hinh

Dung khi muon so sanh nhieu model va kich thuoc anh:

```powershell
python train_sweep.py `
  --data dataset_documents_multisig_split/data.yaml `
  --models yolo11n.pt,yolo11s.pt,yolo11l.pt `
  --imgszs 1024,1280 `
  --epochs 100 `
  --batch 4
```

Xem truoc lenh ma khong train:

```powershell
python train_sweep.py --data dataset_documents_multisig_split/data.yaml --dry_run
```

## 8. Danh gia model

Danh gia Precision/Recall/mAP rieng cho `signature` va `stamp`:

```powershell
python eval_per_class.py `
  --weights runs_signature/yolo11_sig_stamp/weights/best.pt `
  --data dataset_documents_multisig_split/data.yaml `
  --imgsz 1024 `
  --conf 0.15
```

Phan tich loi false positive/false negative:

```powershell
python analyze_errors.py `
  --weights runs_signature/yolo11_sig_stamp/weights/best.pt `
  --data dataset_documents_multisig_split/data.yaml `
  --split val `
  --imgsz 1024 `
  --conf 0.15 `
  --out_csv error_analysis.csv `
  --save_crops error_crops
```

## 9. Predict mot anh

Lenh co ban:

```powershell
python predict_single_image.py `
  --weights runs_signature/yolo11_sig_stamp/weights/best.pt `
  --image test/image-copy-2.png `
  --conf 0.15 `
  --imgsz 1024
```

Output mac dinh:

- `prediction_result.jpg`: anh da ve bounding box
- `prediction_boxes.json`: toa do box theo pixel anh goc
- `prediction_crops/`: crop tung chu ky/con dau phat hien duoc

Neu anh scan lon va chu ky nho, nen dung tiling:

```powershell
python predict_single_image.py `
  --weights runs_signature/yolo11_sig_stamp/weights/best.pt `
  --image test/IMG.jpg `
  --conf 0.15 `
  --imgsz 1280 `
  --tile_size 1280 `
  --tile_overlap 0.25
```

Neu biet chu ky/con dau chi nam o cuoi trang:

```powershell
python predict_single_image.py `
  --weights runs_signature/yolo11_sig_stamp/weights/best.pt `
  --image test/IMG.jpg `
  --conf 0.15 `
  --imgsz 1280 `
  --roi bottom
```

## 10. Goi y tham so

- `--conf 0.10` den `0.15`: uu tien recall, giam nguy co bo sot chu ky
- `--imgsz 1024` hoac `1280`: phu hop voi chu ky/con dau nho tren anh chung tu
- `--batch`: giam neu bi het VRAM
- `--device cpu`: dung khi khong co GPU
- `--tile_size`: nen bat khi anh dau vao rat lon
- `--crop_pad`: them le quanh crop neu muon hau xu ly de hon

## 11. Luong chay nhanh tu dau den cuoi

```powershell
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt

python generate_dataset.py --num_samples 2000 --out_dir dataset_documents_multisig --documents_dir documents --signatures_dir signatures_dir --regions_json document_regions.json
python split_dataset.py --src_dir dataset_documents_multisig --dst_dir dataset_documents_multisig_split --val_ratio 0.15
python train.py --data dataset_documents_multisig_split/data.yaml --model yolo11s.pt --epochs 100 --imgsz 1024 --batch 4 --device 0
python eval_per_class.py --weights runs_signature/yolo11_sig_stamp/weights/best.pt --data dataset_documents_multisig_split/data.yaml --imgsz 1024 --conf 0.15
python predict_single_image.py --weights runs_signature/yolo11_sig_stamp/weights/best.pt --image test/image-copy-2.png --conf 0.15 --imgsz 1024
```

## 12. Loi thuong gap

`torch.cuda.is_available()` tra ve `False`:

- May chua co GPU NVIDIA, driver/CUDA chua dung, hoac PyTorch dang la ban CPU.
- Co the train tam bang `--device cpu`, nhung se cham hon nhieu.

Bi het VRAM khi train:

- Giam `--batch`, vi du tu `4` xuong `2` hoac `1`.
- Giam `--imgsz` neu can.

Model bo sot chu ky:

- Giam `--conf` khi predict, vi du `0.15` hoac `0.10`.
- Tang anh co chu ky nho/bi con dau che trong dataset.
- Dung `--imgsz 1280` hoac tiling voi anh scan lon.

Nhieu box trung nhau khi dung tiling:

- Tang `--nms_iou` hoac dieu chinh `--tile_overlap`.
- Kiem tra lai nguong `--conf`.
