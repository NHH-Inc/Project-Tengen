"""Check the dedicated Windows/CUDA environment for the automatic dataset pipeline."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _module(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}
    return {"ok": True, "version": getattr(module, "__version__", "unknown")}


def check_environment(model_config: str | None = None, checkpoint: str | None = None) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "python": {
            "ok": sys.version_info[:2] == (3, 11),
            "version": ".".join(str(value) for value in sys.version_info[:3]),
            "executable": sys.executable,
        },
        "opencv": _module("cv2"),
        "ultralytics": _module("ultralytics"),
        "groundingdino": _module("groundingdino"),
    }
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        checks["nvidia_smi"] = {"ok": result.returncode == 0, "detail": result.stdout.strip() or result.stderr.strip()}
    else:
        checks["nvidia_smi"] = {"ok": False, "detail": "nvidia-smi is not on PATH"}
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        checks["pytorch"] = {
            "ok": cuda,
            "version": torch.__version__,
            "cuda_available": cuda,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if cuda else None,
        }
    except Exception as exc:
        checks["pytorch"] = {"ok": False, "detail": str(exc)}
    if model_config:
        checks["grounding_dino_model_config"] = {"ok": Path(model_config).is_file(), "path": model_config}
    if checkpoint:
        checks["grounding_dino_checkpoint"] = {"ok": Path(checkpoint).is_file(), "path": checkpoint}
    return {"ok": all(bool(item.get("ok")) for item in checks.values()), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    report = check_environment(args.model_config, args.checkpoint)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
