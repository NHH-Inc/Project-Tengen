"""Measure actual native presentation while robot/shot/heat-map overlays are enabled.

python -m desktop.benchmark --run data/desktop/runs/<id> [--video original-60fps.mp4]
"""
import argparse
import json
from pathlib import Path
import time

from PySide6.QtWidgets import QApplication

from desktop.app import MainWindow, STYLE
from desktop.core import RUNS, probe_video, write_json
from desktop.verify import wait


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--output", type=Path, default=RUNS.parent / "verification/playback.json")
    args = parser.parse_args()
    app = QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    errors = []
    window.error = lambda message: errors.append(str(message))
    try:
        window.open_review(args.run)
        assert window.review is not None and not errors, errors
        if args.video:
            window.open_video(args.video, clear=False)
        video = Path(window.source)
        metadata = probe_video(video)
        # Keep the heat map visible during the test, including its live markers.
        heat_index = next(i for i in range(window.tabs.count()) if window.tabs.tabText(i) == "Heat map")
        window.tabs.setCurrentIndex(heat_index)
        window.audio.setMuted(True)
        window.player.play()
        wait(app, lambda: window.canvas.frames_received >= 5, 15)
        window.player.setPosition(60000)
        wait(app, lambda: window.canvas.time >= 61.0, 10)
        wall_start = time.monotonic()
        frame_start = window.canvas.frames_received
        paint_start = window.canvas.painted_frames
        media_start = window.canvas.time
        while time.monotonic() - wall_start < args.seconds:
            app.processEvents()
            time.sleep(.001)
        elapsed = time.monotonic() - wall_start
        frames = window.canvas.frames_received - frame_start
        painted = window.canvas.painted_frames - paint_start
        advance = window.canvas.time - media_start
        samples = [sample for sample in window.canvas.frame_times if sample[0] >= wall_start]
        intervals = [b[0] - a[0] for a, b in zip(samples, samples[1:])]
        window.player.pause()
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        screenshot = output.with_suffix(".png")
        window.grab().save(str(screenshot))
        report = dict(source=str(video), source_fps=metadata["fps"], wall_seconds=elapsed,
                      presented_frames=frames, painted_frames=painted, measured_fps=frames / elapsed,
                      painted_fps=painted / elapsed,
                      playback_speed=advance / elapsed,
                      p95_frame_interval_ms=sorted(intervals)[int(.95 * (len(intervals) - 1))] * 1000,
                      tracks=len(window.tracks), shots=len(window.canvas.shots),
                      goals=len(window.canvas.goals), heat_samples=len(window.heatmap.samples),
                      cropped=window.canvas.crop_enabled, errors=errors, screenshot=str(screenshot))
        report["passed"] = min(frames, painted) / elapsed >= min(metadata["fps"], 60) * .9 and .9 <= advance / elapsed <= 1.1 and not errors
        write_json(output, report)
        print(json.dumps(report, indent=2))
        assert report["passed"], "Playback fell below 90% of native frame rate"
    finally:
        window.close()
        app.processEvents()


if __name__ == "__main__":
    main()
