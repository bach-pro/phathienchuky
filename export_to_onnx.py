from argparse import ArgumentParser
from pathlib import Path
from ultralytics import YOLO


DEFAULT_MODEL = Path(
    r"runs\detect\runs_signature\yolo26_sig_stamp\weights\best.pt"
)


def export_to_onnx(model_path: Path, imgsz: int, opset: int | None) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
    )
    return Path(exported_path)


def main() -> None:
    parser = ArgumentParser(description="Export a YOLO .pt model to .onnx")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to .pt model. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Export image size. Use the same size used for training/validation.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=20,
        help="ONNX opset version.",
    )
    args = parser.parse_args()

    onnx_path = export_to_onnx(args.model, args.imgsz, args.opset)
    print(f"Exported ONNX model: {onnx_path}")


if __name__ == "__main__":
    main()
