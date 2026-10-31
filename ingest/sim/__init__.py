"""Tengen's field simulation (PLAN-2026 section 2; design in docs/SIM-DESIGN.md).

A deterministic, headless motion model for robots on the field, plus a validation harness that
checks the model against real tracks. This is the Python core; the interactive web view drives it.

Tiers (docs/SIM-DESIGN.md section 3), built up over phases:
  T0  kinematic replay + validation harness   (this package: validate.py)
  T1  motion dynamics: velocity/accel limits   (this package: model.py)
  T2  robot-robot interactions / defense       (later)
  T3  game pieces + scoring                     (later)
"""

from .model import Pose, RobotParams, params_from_profile, simulate_path
from .validate import auto_waypoints, trajectory_error, validate_auto

__all__ = [
    "Pose", "RobotParams", "params_from_profile", "simulate_path",
    "auto_waypoints", "trajectory_error", "validate_auto",
]
