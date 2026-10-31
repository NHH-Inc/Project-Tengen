"""Tengen's field simulation (PLAN-2026 section 2; design in docs/SIM-DESIGN.md).

A deterministic, headless motion model for robots on the field, plus a validation harness that
checks the model against real tracks. This is the Python core; the interactive web view drives it.

Tiers (docs/SIM-DESIGN.md section 3), built up over phases:
  T0  kinematic replay + validation harness   (validate.py)
  T1  motion dynamics: velocity/accel limits   (model.py)
  T2  robot-robot interactions / defense       (match.py; turn-rate limit in model.py)
  T3  game pieces + scoring                     (game.py, scoring.py; rules from the season config)
"""

from .game import GameModel, Goal, load_game_model
from .match import SimRobot, reached, simulate_match
from .model import Pose, RobotParams, params_from_profile, simulate_path
from .scoring import Shooter, score_tracks, shooter_from_profile
from .validate import auto_waypoints, trajectory_error, validate_auto

__all__ = [
    "Pose", "RobotParams", "params_from_profile", "simulate_path",
    "SimRobot", "simulate_match", "reached",
    "GameModel", "Goal", "load_game_model",
    "Shooter", "score_tracks", "shooter_from_profile",
    "auto_waypoints", "trajectory_error", "validate_auto",
]
