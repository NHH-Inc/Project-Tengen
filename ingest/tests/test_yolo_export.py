import json
import tempfile
import unittest
from pathlib import Path

from ingest.collection.config import load_config
from ingest.collection.yolo_export import export_yolo_dataset


class YoloExportTests(unittest.TestCase):
    def _config(self, root: Path):
        path = root / "config.yaml"
        path.write_text(
            """season: 2026
game: REBUILT
storage: {root: data, segments: data/segments, collections: data/collections, datasets: data/datasets}
sampling: {fps: 1}
split: {seed: 1, train: 0.7, val: 0.15, test: 0.15, group_by: event}
classes: [{name: robot, id: 0}]
""",
            encoding="utf-8",
        )
        return load_config(path)

    def test_export_writes_yolo_boxes_and_explicit_negative_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection = root / "collection"
            frames_dir = collection / "frames" / "match"
            frames_dir.mkdir(parents=True)
            for name in ("one.jpg", "two.jpg"):
                (frames_dir / name).write_bytes(b"image")
            frames = [
                {"frame_id": "one", "image_path": "frames/match/one.jpg", "split": "train", "event_id": "2026a", "split_group": "2026a"},
                {"frame_id": "two", "image_path": "frames/match/two.jpg", "split": "train", "event_id": "2026a", "split_group": "2026a"},
            ]
            labels = [
                {"frame_id": "one", "status": "proposed", "boxes": [{"class_name": "robot", "x": .1, "y": .2, "w": .3, "h": .4}]},
                {"frame_id": "two", "status": "proposed", "boxes": []},
            ]
            (collection / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in frames), encoding="utf-8")
            (collection / "filtered-proposals.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8")
            output = root / "dataset"
            report = export_yolo_dataset(
                collection=collection, config=self._config(root), output=output, allow_unreviewed=True,
            )
            self.assertEqual((output / "labels" / "train" / "one.txt").read_text().strip(), "0 0.250000 0.400000 0.300000 0.400000")
            self.assertEqual((output / "labels" / "train" / "two.txt").read_text(), "")
            self.assertEqual(report["splits"]["train"]["negative_images"], 1)

    def test_export_rejects_event_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection = root / "collection"
            frames_dir = collection / "frames"
            frames_dir.mkdir(parents=True)
            (frames_dir / "one.jpg").write_bytes(b"image")
            (frames_dir / "two.jpg").write_bytes(b"image")
            frames = [
                {"frame_id": "one", "image_path": "frames/one.jpg", "split": "train", "split_group": "2026a"},
                {"frame_id": "two", "image_path": "frames/two.jpg", "split": "val", "split_group": "2026a"},
            ]
            labels = [{"frame_id": name, "status": "proposed", "boxes": []} for name in ("one", "two")]
            (collection / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in frames), encoding="utf-8")
            (collection / "filtered-proposals.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Split leakage"):
                export_yolo_dataset(
                    collection=collection, config=self._config(root), output=root / "dataset",
                    allow_unreviewed=True,
                )


if __name__ == "__main__":
    unittest.main()
