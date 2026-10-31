"""The game model (T3): goals, game pieces and point values, read from the season config.

Everything game-specific lives in `contracts/seasons/<year>.json` and the strategy geometry, so the
sim stays game-agnostic and the 2026 numbers slot in without code changes. Today those point values
are placeholders (zeros) -- the framework is built and tested; fill in the real values and scoring
comes alive. See docs/SIM-DESIGN.md section 3 (T3).

A team scores in the **opponent** hub (we are in the alliance frame; see contracts/strategy/<year>),
so every goal sits at the opponent-hub position. 'high' and 'low' are the same hub at different
heights -- a 2D strategy sim treats them as one place with different point values.
"""

from __future__ import annotations

from dataclasses import dataclass

#: A shooter within this of the hub can score, unless the season config overrides it. A placeholder
#: like the point values: a real shooting-range number for the game replaces it in the config.
DEFAULT_SHOT_RANGE_FT = 15.0
_PHASES = ("auto", "teleop", "endgame")


@dataclass(frozen=True)
class Goal:
    name: str
    x: float
    y: float
    effective_range_ft: float
    points: dict  # phase -> points per made shot


@dataclass(frozen=True)
class GameModel:
    pieces: tuple[str, ...]
    goals: tuple[Goal, ...]
    auto_seconds: float
    teleop_seconds: float  # includes the endgame (Doc 0)
    endgame_seconds: float

    def goal(self, name: str) -> Goal | None:
        return next((g for g in self.goals if g.name == name), None)

    def phase_at(self, t: float) -> str:
        """The match phase at time `t` (seconds from match start). Teleop contains the endgame, so
        the endgame is its final `endgame_seconds`."""
        if t < self.auto_seconds:
            return "auto"
        if t < self.auto_seconds + self.teleop_seconds - self.endgame_seconds:
            return "teleop"
        return "endgame"

    @property
    def match_seconds(self) -> float:
        return self.auto_seconds + self.teleop_seconds


def load_game_model(season: dict, geometry: dict, *, shot_range_ft: float | None = None) -> GameModel:
    """Build the game model from a season config (`contracts/seasons/<year>.json`) and the strategy
    geometry (for the hub position). Missing point values default to 0, so a placeholder config
    yields a valid, zero-scoring game rather than an error."""
    hub = geometry["hubs"]["opponent_hub"]
    point_values = season.get("point_values", {})
    rng = shot_range_ft if shot_range_ft is not None else float(season.get("shot_range_ft", DEFAULT_SHOT_RANGE_FT))
    goals = tuple(
        Goal(name=name, x=float(hub["x"]), y=float(hub["y"]), effective_range_ft=rng,
             points={phase: float(point_values.get(phase, {}).get(f"shot_made_{name}", 0))
                     for phase in _PHASES})
        for name in season.get("goals", [])
    )
    return GameModel(
        pieces=tuple(season.get("game_pieces", [])),
        goals=goals,
        auto_seconds=float(season.get("auto_seconds", 0)),
        teleop_seconds=float(season.get("teleop_seconds", 0)),
        endgame_seconds=float(season.get("endgame_seconds", 0)),
    )
