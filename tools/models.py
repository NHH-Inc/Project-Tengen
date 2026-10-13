"""The model registry: every robot detector, numbered, with its scores.

    python -m tools.models list                 # the table, from training/registry
    python -m tools.models evaluate [--only v3] # score every model whose file is here
    python -m tools.models markdown             # write docs/MODELS.md

Facts about each model (who trained it, on what, its own val numbers) live in
training/registry/models.json. Held-out scores are generated into training/registry/scores/ by
`evaluate` and never typed in: a number a person wrote down is a number nobody can reproduce.

Why two kinds of score. Each model's val split is different -- v1, v2 and v3 were validated on
different images -- so val mAP does not compare across versions. The held-out pack is the same
400 frames of real 2026 footage for every model, which is the only comparison that means
anything, and its headline is the share of frames where the model finds all six robots.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "training" / "registry"
SCORES = REGISTRY / "scores"


def load() -> dict:
    return json.loads((REGISTRY / "models.json").read_text(encoding="utf-8"))


def score(model_id: str) -> dict | None:
    path = SCORES / f"{model_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def models_with_scores() -> list[dict]:
    """Registry entries, oldest first, each with its held-out score attached (or None)."""
    return [{**m, "heldout": score(m["id"])} for m in load()["models"]]


def evaluate(only: set[str] | None = None) -> None:
    from ingest.collection.detect_runner import OnnxDetector
    from ingest.collection.viewpoint_eval import evaluate as run

    registry = load()
    pack = ROOT / registry["yardstick"]["pack"]
    SCORES.mkdir(parents=True, exist_ok=True)
    for model in registry["models"]:
        if only and model["id"] not in only:
            continue
        path = ROOT / model["file"]
        if not path.is_file():
            print(f"{model['id']:8} skipped: {model['file']} is not on this machine")
            continue
        detector = OnnxDetector(model_path=str(path), confidence_threshold=0.25)
        result = run(detector, pack, expected=6)
        frames = result["frames"]
        out = {
            "model": model["id"],
            "file": model["file"],
            "input_px": detector.input_size,
            "threshold": 0.25,
            "pack": registry["yardstick"]["pack"],
            "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "frames": frames,
            "mean_robots": round(result["mean"], 3),
            "blind_frames": result["blind_frames"],
            "full_frames": result["full_frames"],
            "full_share": round(result["full_frames"] / frames, 4),
            "histogram": {str(k): v for k, v in sorted(result["histogram"].items())},
            "per_frame": result["per_frame"],
        }
        (SCORES / f"{model['id']}.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"{model['id']:8} {out['full_share']:.1%} of frames see all six, mean {out['mean_robots']:.2f}, "
              f"{out['blind_frames']} blind")


def _pct(value) -> str:
    return "—" if value is None else f"{value:.1%}"


def table() -> list[list[str]]:
    rows = []
    for m in reversed(models_with_scores()):
        h = m["heldout"] or {}
        pipeline = (m.get("pipeline") or {}).get("all_six_tracked_share")
        rows.append([
            f"**{m['id']}**" + (" ⭐" if m["status"] == "current" else ""),
            m["status"],
            m["trained_on"],
            m["trained_by"],
            f"{m['architecture']} @ {m['train_size_px']}",
            _pct(h.get("full_share")),
            "—" if not h else f"{h['mean_robots']:.2f}",
            "—" if not h else str(h["blind_frames"]),
            _pct(pipeline),
            f"{m['val']['map50']:.3f} / {m['val']['map50_95']:.3f}",
        ])
    return rows


HEADER = ["Model", "Status", "Trained", "By", "Network", "Held-out: all 6 found",
          "Mean robots/frame", "Blind frames", "Match: all 6 tracked", "Own val mAP50 / 50-95"]


def markdown() -> str:
    lines = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    lines += ["| " + " | ".join(row) + " |" for row in table()]
    return "\n".join(lines)


def write_models_doc() -> Path:
    registry = load()
    path = ROOT / "docs" / "MODELS.md"
    details = []
    for m in reversed(models_with_scores()):
        h = m["heldout"]
        details.append(f"### {m['id']} — {m['name']}" + (" (current)" if m["status"] == "current" else ""))
        details.append("")
        details.append(f"- **Trained** {m['trained_on']} by {m['trained_by']}: {m['architecture']}, "
                       f"{m['epochs']} epochs at {m['train_size_px']} px.")
        details.append(f"- **Data** {m['dataset']}.")
        v = m["val"]
        details.append(f"- **Its own val split** P {v['precision']:.3f}, R {v['recall']:.3f}, "
                       f"mAP50 {v['map50']:.3f}, mAP50-95 {v['map50_95']:.3f}.")
        if h:
            details.append(f"- **Held-out** ({h['frames']} frames, {h['input_px']} px input, evaluated "
                           f"{h['evaluated_at'][:10]}): all six robots in {h['full_share']:.1%} of frames, "
                           f"mean {h['mean_robots']:.2f}, {h['blind_frames']} blind.")
        if m.get("pipeline"):
            p = m["pipeline"]
            details.append(f"- **Whole pipeline** on {p['match']}: all six tracked {p['all_six_tracked_share']:.0%} "
                           f"of the match, at least five {p['at_least_five_tracked_share']:.0%}. {p['notes']}")
        if m.get("run"):
            details.append(f"- **Training curves** [`{Path(m['run']).name}`](../{m['run']})")
        details.append(f"- {m['notes']}")
        details.append("")
    text = f"""# Models

Every robot detector Tengen has had, numbered. Generated by `python -m tools.models markdown`
from [`training/registry/`](../training/registry/); edit the registry, not this file.

**How to read it.** The held-out columns are the same {registry['yardstick']['pack'].split('/')[-1]}
pack of 400 real 2026 broadcast frames for every model, so they compare across versions. Each
model's own val mAP was measured on its own val split, so it does **not** compare across versions,
and a higher val mAP has more than once come with a worse model on real footage. "Match: all 6
tracked" measures the whole pipeline (detector plus tracking and identity) on one real match.

{markdown()}

![status](status/status.svg)

{chr(10).join(details)}
## Adding a model

See [IMPROVING-THE-MODEL.md](IMPROVING-THE-MODEL.md). In short: add an entry to
`training/registry/models.json` with the next number, put its ONNX in `data/`, run
`python -m tools.models evaluate --only vN`, then `python -m tools.status` and
`python -m tools.models markdown`. It becomes `current` only if it beats the current model on
every held-out pack.
"""
    path.write_text(text, encoding="utf-8")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.models", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["list", "evaluate", "markdown"])
    parser.add_argument("--only", nargs="*", help="model ids to evaluate")
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        evaluate(set(args.only) if args.only else None)
    elif args.command == "markdown":
        print(f"wrote {write_models_doc().relative_to(ROOT)}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
