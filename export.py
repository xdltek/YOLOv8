#!/usr/bin/env python3
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download YOLOv8 weights if needed and export them to ONNX."
    )
    parser.add_argument(
        "--weights",
        default="yolov8m.pt",
        help="Ultralytics weights name or local path, for example yolov8n.pt or yolov8m.pt",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[640],
        help="Image size. Use one value for square input, or two values for height width",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for export",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Export device, for example cpu or 0",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export FP16 model when supported",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic input shape",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify exported ONNX model",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory used to store the exported ONNX model",
    )
    return parser.parse_args()


def normalize_imgsz(imgsz):
    if len(imgsz) == 1:
        return imgsz[0]
    if len(imgsz) == 2:
        return imgsz
    raise ValueError("--imgsz accepts either one value or two values")


def main():
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=normalize_imgsz(args.imgsz),
            opset=args.opset,
            batch=args.batch,
            device=args.device,
            half=args.half,
            dynamic=args.dynamic,
            simplify=args.simplify,
        )
    ).resolve()

    target_path = output_dir / exported_path.name
    if exported_path != target_path:
        exported_path.replace(target_path)

    print(f"ONNX export complete: {target_path}")


if __name__ == "__main__":
    main()
