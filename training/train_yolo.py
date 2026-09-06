"""Train an FRC robot detector with Ultralytics YOLO11 on a YOLO-format dataset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="dataset.yaml produced by export-yolo")
    parser.add_argument("--output", required=True, help="new output directory; never overwritten")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--batch", default="-1", help="integer batch, -1 for auto, or 0..1 GPU fraction")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = Path(args.data).resolve()
    if not data.is_file():
        raise SystemExit(f"Dataset YAML is missing: {data}")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty model directory: {output}")
    if args.epochs <= 0 or args.image_size <= 0 or args.image_size % 32:
        raise SystemExit("epochs must be positive and image-size must be a positive multiple of 32")
    try:
        batch = float(args.batch) if "." in str(args.batch) else int(args.batch)
    except ValueError as exc:
        raise SystemExit("batch must be an integer, -1, or a 0..1 GPU fraction") from exc
    if isinstance(batch, float) and not 0 < batch <= 1:
        raise SystemExit("a fractional batch must be in (0, 1]")
    if isinstance(batch, int) and batch != -1 and batch <= 0:
        raise SystemExit("an integer batch must be positive or -1 for auto")
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install training/requirements-yolo.txt in the dedicated vision venv") from exc
    if str(args.device) != "cpu" and not torch.cuda.is_available():
        raise SystemExit("YOLO training requested a GPU, but CUDA is not available")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.train(
        data=str(data), epochs=args.epochs, imgsz=args.image_size, batch=batch,
        device=args.device, workers=args.workers, patience=args.patience,
        project=str(output.parent), name=output.name, exist_ok=False,
    )
    best = output / "weights" / "best.pt"
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": args.model,
        "data": str(data),
        "epochs": args.epochs,
        "image_size": args.image_size,
        "batch": batch,
        "device": args.device,
        "best_checkpoint": str(best),
    }
    (output / "training-config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
