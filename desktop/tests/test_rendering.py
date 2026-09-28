import json
from pathlib import Path

import pytest
from PySide6.QtCore import QSizeF
from PySide6.QtWidgets import QApplication

from desktop.core import FIXTURE, Review, box_at
from desktop.rendering import HeatMap, VideoCanvas, scoring_counts, shot_path_at, goal_counting_paused


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_crop_uses_model_coordinates_and_removes_bottom_panel(app):
    canvas = VideoCanvas()
    canvas.set_source_size(QSizeF(1920, 1080))
    assert canvas.clip.rect().width() == pytest.approx(1843.2)
    assert canvas.clip.rect().height() == pytest.approx(675)
    assert canvas.video_item.pos().x() == pytest.approx(-38.4)
    assert canvas.video_item.pos().y() == pytest.approx(-37.8)
    assert canvas.point(0, 0).x() == pytest.approx(0)
    assert canvas.point(0, 0).y() == pytest.approx(0)
    assert canvas.point(1, 1).y() == pytest.approx(675)
    canvas.crop_enabled = False
    canvas.update_geometry()
    assert canvas.point(0, 0).x() == pytest.approx(38.4)
    assert canvas.point(0, 0).y() == pytest.approx(37.8)
    assert canvas.clip.rect().height() == 1080
    canvas.close()


def test_shots_and_goal_entries_are_time_scoped_and_not_double_counted():
    shots = [dict(shot_id="one", launch_t_seconds=1, outcome="made", outcome_t_seconds=2, robot_track_id=3),
             dict(shot_id="two", launch_t_seconds=3, outcome="made", outcome_t_seconds=4, robot_track_id=None)]
    entries = [dict(shot_id="one", t_seconds=2, robot_track_id=3)]
    assert scoring_counts(shots, entries, 1.9) == (0, 0)
    assert scoring_counts(shots, entries, 2.1) == (1, 0)
    assert scoring_counts(shots, entries, 5) == (2, 1)
    shot = dict(launch_t_seconds=1, ball_track=[dict(t_seconds=t, x=0, y=0) for t in (.9, 1, 1.1, 1.5)])
    assert [p["t_seconds"] for p in shot_path_at(shot, 1.1)] == [.9, 1, 1.1]
    assert shot_path_at(shot, 1.8) == []
    assert goal_counting_paused([[2, 4], [8, None]], 3)
    assert not goal_counting_paused([[2, 4], [8, None]], 4)
    assert goal_counting_paused([[2, 4], [8, None]], 9)


def test_heatmap_never_leaks_future_positions_and_supports_filtering(app):
    heat = HeatMap()
    def box(t, x):
        return dict(t=t, x=0, y=0, w=.1, h=.1, field_x=x, field_y=5)
    tracks = [dict(track_id=1, team=123, boxes=[box(0, 1), box(1, 2), box(5, 3)], gaps=[]),
              dict(track_id=2, team=456, boxes=[box(0, 4), box(1, 5), box(5, 6)], gaps=[])]
    heat.set_data(tracks, dict(field_length_ft=54, field_width_ft=26.6))
    heat.set_time(1.1)
    assert len(heat.visible_samples) == 4
    assert not heat.grid_image.isNull()
    heat.set_time(.1)
    assert len(heat.visible_samples) == 2
    heat.selected_team = 123
    heat.rebuild_samples()
    assert len(heat.visible_samples) == 1
    assert all(p[3]["team"] == 123 for p in heat.visible_samples)
    assert box_at(tracks[0], .5)["field_x"] == pytest.approx(1.5)
    heat.close()


def test_review_loads_shot_calibration_entries_and_exports_them(tmp_path):
    (tmp_path / "job.json").write_text((FIXTURE / "job.json").read_text())
    (tmp_path / "shots.jsonl").write_text('{"shot_id":"one"}\n')
    (tmp_path / "goal_entries.jsonl").write_text('{"entry_id":"entry"}\n')
    (tmp_path / "ball_scouting.config.json").write_text(json.dumps(dict(goals=[dict(id="high")], goal_calibration=dict(camera_gaps=[[2, 4]]))))
    review = Review(tmp_path)
    assert review.shots_configured
    assert review.goals == [dict(id="high")]
    assert review.goal_calibration["camera_gaps"] == [[2, 4]]
    review.export(tmp_path / "export.json")
    exported = json.loads((tmp_path / "export.json").read_text())
    assert exported["goal_entries"] == [dict(entry_id="entry")]
    assert exported["goals"] == review.goals
