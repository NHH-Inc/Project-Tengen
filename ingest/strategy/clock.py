"""Match time from the scoreboard clock, fitted rather than trusted reading by reading.

A clip is not a match. On a real 2026 Championship upload the title card runs 3 s, the clock sits
frozen at 0:20 until 7.3 s, auto runs 20 s, the clock freezes again at 2:20 for about 4 s before
teleop, and the match ends with 40 s of video still to go. Phases computed from "the clip starts
at match time zero" are wrong by seconds everywhere, and one fixed offset cannot represent the
pause between auto and teleop at all.

Single OCR readings are not trustworthy either. On that same match "2:20" read as "22" twice and
the frozen "0:20" once read as "29". So nothing here believes one reading. While a period runs,
the clock falls one second per second, so every reading votes for when its period started:
`start = t - (length - remaining)`. Correct readings agree to within a second; misreads and frozen
clocks scatter. The period with the most agreeing votes wins, and the votes that agreed are
reported, so a fit can be checked.

Period lengths are measured from the clock: from where its countdown begins to where it ends,
snapped to a whole second (and to the season file's value when within a second). When the clock
and the season file disagree, the measured length is used and the disagreement is reported,
because the clock is what the referees ran.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import median

#: Votes within this many seconds of each other agree.
AGREE_S = 1.5
#: A period needs at least this many agreeing readings to count as found.
MIN_SUPPORT = 4
#: A value held for this many consecutive readings is a frozen clock showing a period length.
FROZEN_READINGS = 3
#: The longest plausible pause between auto ending and teleop starting.
MAX_PAUSE_S = 12.0


@dataclass
class MatchClock:
    """Maps video seconds to match seconds and phase.

    `mt` is match play elapsed: 0 to auto length during auto, held at the auto length through the
    pause, then counting on through teleop. A clock with nothing found falls back to a fixed start.
    """
    auto_start_t: float | None
    auto_seconds: float
    teleop_start_t: float | None
    teleop_seconds: float
    endgame_seconds: float
    source: str
    support: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def match_start_t(self) -> float:
        """Video seconds of match time zero."""
        if self.auto_start_t is not None:
            return self.auto_start_t
        if self.teleop_start_t is not None:
            # Clip began during the pause or teleop: the pause cannot be measured, so assume none.
            return self.teleop_start_t - self.auto_seconds
        return 0.0

    def at(self, t: float) -> tuple[float, str]:
        a, tp = self.match_start_t, self.teleop_start_t
        if t < a:
            return round(t - a, 3), "pre_match"
        if t <= a + self.auto_seconds:
            return round(t - a, 3), "auto"
        teleop_start = tp if tp is not None else a + self.auto_seconds
        if t < teleop_start:
            return round(self.auto_seconds, 3), "transition"
        into = t - teleop_start
        if into > self.teleop_seconds:
            return round(self.auto_seconds + self.teleop_seconds, 3), "post_match"
        phase = "endgame" if self.teleop_seconds - into <= self.endgame_seconds else "teleop"
        return round(self.auto_seconds + into, 3), phase

    def as_dict(self) -> dict:
        return {"auto_start_t": self.auto_start_t, "auto_seconds": self.auto_seconds,
                "teleop_start_t": self.teleop_start_t, "teleop_seconds": self.teleop_seconds,
                "endgame_seconds": self.endgame_seconds, "source": self.source,
                "support": self.support, "notes": self.notes}

    @classmethod
    def from_dict(cls, value: dict) -> "MatchClock":
        return cls(**{k: value.get(k) for k in ("auto_start_t", "auto_seconds", "teleop_start_t",
                                                 "teleop_seconds", "endgame_seconds", "source")},
                   support=value.get("support") or {}, notes=value.get("notes") or [])


def season_lengths(season: dict) -> tuple[float, float, float]:
    """(auto, teleop, endgame) as the clock shows them.

    Doc 0: `teleop_seconds` includes the endgame, which is its last `endgame_seconds`; the clock
    likewise runs one teleop period with the endgame at its end.
    """
    return (float(season.get("auto_seconds", 0)), float(season.get("teleop_seconds", 0)),
            float(season.get("endgame_seconds", 0)))


def frozen_values(readings: list[tuple[float, int | None]]) -> set[int]:
    """Clock values held unchanged for several consecutive readings, other than zero."""
    values, run, previous = set(), 0, None
    for _, value in readings:
        run = run + 1 if value is not None and value == previous else 1
        previous = value
        if value and run >= FROZEN_READINGS:
            values.add(value)
    return values


def _best_end(readings, window=None):
    """The period end most readings agree on, and the readings that agree.

    While a period runs, `t + remaining` is the same for every correct reading whatever the
    period's length, which is why the END is voted on: voting on the start needs the length, and
    a wrong length shifts every vote together, so a wrong guess looks exactly as consistent.
    """
    votes = [(t + value, t, value) for t, value in readings if value is not None and value > 0]
    if window:
        votes = [v for v in votes if window[0] <= v[0] <= window[1]]
    best, support = None, []
    for candidate, _, _ in votes:
        agreeing = [v for v in votes if abs(v[0] - candidate) <= AGREE_S]
        if len(agreeing) > len(support):
            best, support = candidate, agreeing
    support = _running(sorted(support, key=lambda v: v[1]), dict(readings))
    if best is None or len(support) < MIN_SUPPORT:
        return None, []
    return median(v[0] for v in support), support


def _running(support, values_at):
    """Drop leading votes that are not a running clock.

    A frozen clock, or a misread just before the countdown, can land within a second of the right
    end by coincidence. A running clock never shows one value twice in a row, and falls by the
    seconds elapsed. Trimming those votes is what keeps the measured start (and so the length)
    from being dragged earlier: on a real Istanbul match, two misread "23"s and a frozen "20"
    agreed with auto's end and stretched a 20 s auto to 155 s.
    """
    times = sorted(values_at)
    position = {t: i for i, t in enumerate(times)}

    def frozen_at(t, value):
        i = position[t]
        return i + 1 >= FROZEN_READINGS and all(
            values_at[times[i - k]] == value for k in range(1, FROZEN_READINGS))

    while len(support) >= 2:
        (_, t0, v0), (_, t1, v1) = support[0], support[1]
        frozen = frozen_at(t0, v0)
        if not frozen and v0 != v1 and abs((v0 - v1) - (t1 - t0)) <= 0.6:
            break
        support = support[1:]
    return support


def _length(end, support, readings, candidates, default):
    """How long the period was, from the window its countdown must have started in.

    A frozen clock (one value read three times running) has not started; the first agreeing reading
    has. So the start lies after the last frozen reading and at or before the first agreeing one,
    which bounds the length to a few whole seconds. The season file's length wins if it fits that
    window; then a value the clock was seen frozen at; otherwise the window's middle, rounded.

    With no frozen reading before the countdown -- a clip that starts mid-period -- the start is
    unknown and the length is the season file's (or, if the clock already showed more than that,
    just over the highest value seen), never the first reading we happened to see.
    """
    first_t = min(t for _, t, _ in support)
    top = max(value for _, _, value in support)
    before = [(t, value) for t, value in readings if t < first_t]
    frozen_t = None
    for i in range(FROZEN_READINGS - 1, len(before)):
        window = [value for _, value in before[i - FROZEN_READINGS + 1: i + 1]]
        if window[0] is not None and all(value == window[0] for value in window):
            frozen_t = before[i][0]
    if frozen_t is None:
        return float(default if default >= top else top + 1), "assumed"
    shortest, longest = end - first_t, end - frozen_t
    # One-second sampling and an unknown display rounding leave about a second either way.
    fits = lambda c: shortest - 1.0 <= c <= longest + 1.0 and c >= top
    if fits(default):
        return float(default), "clock"
    known = [c for c in candidates if fits(c)]
    middle = (shortest + longest) / 2
    if known:
        return float(min(known, key=lambda c: abs(c - middle))), "clock"
    return float(round(middle)), "clock"


def fit_clock(readings: list[tuple[float, int | None]], season: dict) -> MatchClock:
    """Fit auto and teleop from (video seconds, clock seconds remaining) readings."""
    auto_len, teleop_len, endgame = season_lengths(season)
    readings = sorted(readings)
    candidates = {auto_len, teleop_len} | frozen_values(readings)
    notes = []

    # The longest run of agreeing readings is teleop, unless it is too short to be teleop (a clip
    # of auto alone), in which case it is auto.
    end, support = _best_end(readings)
    teleop = auto = None
    if end is not None:
        top = max(value for _, _, value in support)
        is_teleop = top > auto_len + 10
        length, how = _length(end, support, readings, candidates, teleop_len if is_teleop else auto_len)
        if is_teleop:
            teleop = (end - length, length, how, support)
        else:
            auto = (end - length, length, how, support)

    if teleop is not None:
        explained = {t for _, t, _ in teleop[3]}
        rest = [r for r in readings if r[0] not in explained]
        # Auto ends at or shortly before teleop starts.
        end, support = _best_end(rest, (teleop[0] - MAX_PAUSE_S, teleop[0] + AGREE_S))
        if end is not None:
            length, how = _length(end, support, rest, candidates - {teleop[1]}, auto_len)
            auto = (end - length, length, how, support)

    for name, found, expected in (("auto", auto, auto_len), ("teleop", teleop, teleop_len)):
        if found and found[2] == "clock" and found[1] != expected:
            notes.append(f"clock shows {name} {found[1]:.0f} s; season file implies {expected:.0f} s")
        if found and found[2] == "assumed":
            notes.append(f"{name} length taken from the season file ({found[1]:.0f} s): no frozen clock "
                         f"was read before its countdown (the clip may start mid-period, or "
                         f"the frozen clock was misread)")
    if auto is None and teleop is None:
        notes.append("no period found on the clock")
    elif auto is None:
        notes.append("auto not found on the clock: assumed it ended as teleop began, with no pause")
    return MatchClock(
        auto_start_t=round(auto[0], 2) if auto else None,
        auto_seconds=auto[1] if auto else auto_len,
        teleop_start_t=round(teleop[0], 2) if teleop else None,
        teleop_seconds=teleop[1] if teleop else teleop_len,
        endgame_seconds=endgame,
        source="scoreboard_clock" if (auto or teleop) else "none",
        support={"auto": len(auto[3]) if auto else 0, "teleop": len(teleop[3]) if teleop else 0,
                 "readings": sum(1 for _, v in readings if v is not None), "samples": len(readings)},
        notes=notes,
    )


def fixed_clock(start_t: float, season: dict, source: str) -> MatchClock:
    """No clock reading: one start time and the season's lengths, with no pause."""
    auto_len, teleop_len, endgame = season_lengths(season)
    return MatchClock(auto_start_t=start_t, auto_seconds=auto_len, teleop_start_t=None,
                      teleop_seconds=teleop_len, endgame_seconds=endgame, source=source)


def read_clock(video_path, step: float = 1.0) -> list[tuple[float, int | None]]:
    """Read the scoreboard clock every `step` seconds, decoding forward without seeking."""
    import cv2

    from ..scoreboard import read_match_timer

    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    wanted = {int(round(i * step * fps)): round(i * step, 3)
              for i in range(int(total / fps / step) + 1)}
    readings, index = [], 0
    last = max(wanted) if wanted else -1
    while index <= last:
        if index in wanted:
            ok, frame = capture.read()
            if not ok:
                break
            readings.append((wanted[index], read_match_timer(frame)))
        elif not capture.grab():
            break
        index += 1
    capture.release()
    return readings


def common_reading(readings) -> int | None:
    values = Counter(v for _, v in readings if v is not None)
    return values.most_common(1)[0][0] if values else None
