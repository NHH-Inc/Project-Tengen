"""Robot-relative paired photogates for rapid-fire bursts.

A short, observed launch seeds a local shooter direction. Two thin sampling strips then
measure yellow occupancy independently of contour/track IDs. Prominent pulses must travel
from the inner strip to the outer strip at a plausible launch speed. We never extrapolate a
shot count from burst duration, yellow area, or an assumed firing frequency.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class LaunchSignalConfig:
    enabled: bool = True
    short_track_minimum_hits: int = 2
    maximum_observation_gap_seconds: float = 0.085
    minimum_travel_radii: float = 1.25
    gate_lifetime_seconds: float = 1.0
    gate_separation_radii: float = 2.5
    pulse_minimum_prominence: float = 0.16
    pulse_relative_prominence: float = 0.22
    maximum_transit_seconds: float = 0.16


@dataclass
class GateSample:
    frame_index: int
    t_seconds: float
    value: float
    center: tuple[float, float]
    radius: float


class ProminentPulseDetector:
    """Causal peak/valley hysteresis; no smoothing that erases one-frame pulses.

    A nonzero valley can rearm the detector, so touching balls need not produce a black
    frame between them. A flat/continuously occupied gate yields at most one pulse. The
    first sample establishes a baseline rather than inventing an edge on camera startup.
    """

    def __init__(self, absolute: float, relative: float):
        self.absolute = absolute
        self.relative = relative
        self.values: deque[float] = deque(maxlen=45)
        self.valley: float | None = None
        self.peak: GateSample | None = None

    def update(self, sample: GateSample) -> GateSample | None:
        history = np.asarray(self.values, dtype=float)
        noise = 0.0
        if len(history) >= 8:
            # Estimate background noise from the lower half, without treating shot peaks
            # as noise and progressively raising the threshold during a burst.
            low = history[history <= np.median(history)]
            noise = min(2 * self.absolute,
                        3.0 * 1.4826 * float(np.median(np.abs(low - np.median(low)))))
        self.values.append(sample.value)
        threshold = max(self.absolute, noise)
        if self.valley is None:
            self.valley = sample.value
            return None
        if self.peak is None:
            self.valley = min(self.valley, sample.value)
            if sample.value - self.valley >= threshold:
                self.peak = sample
            return None
        if sample.value > self.peak.value:
            self.peak = sample
        fall = max(threshold, self.relative * self.peak.value)
        if self.peak.value - sample.value >= fall:
            result = self.peak
            self.peak = None
            self.valley = sample.value
            return result
        return None


@dataclass
class GatePulse:
    robot_id: int
    port_id: int
    inner: GateSample
    outer: GateSample
    speed: float


@dataclass
class _Port:
    port_id: int
    robot_id: int
    anchor: tuple[float, float]
    direction: tuple[float, float]
    radius_ratio: float
    speed: float
    last_evidence: float
    inner: ProminentPulseDetector
    outer: ProminentPulseDetector
    pending: deque[GateSample] = field(default_factory=lambda: deque(maxlen=32))
    previous_box: tuple[float, float, float, float] | None = None
    last_sample_time: float | None = None
    # Observed track evidence and pulse evidence share this geometric crossing registry.
    crossings: deque[tuple[float, float, float, object]] = field(
        default_factory=lambda: deque(maxlen=128)
    )


class LaunchSignalBank:
    def __init__(self, config: LaunchSignalConfig):
        self.config = config
        self.ports: list[_Port] = []
        self.next_port_id = 1
        self.last_samples: list[dict] = []

    @staticmethod
    def geometry(port: _Port, box):
        left, top, right, bottom = box
        scale = max(1.0, math.hypot(right - left, bottom - top))
        radius = max(2.0, port.radius_ratio * scale)
        origin = (left + port.anchor[0] * (right - left),
                  top + port.anchor[1] * (bottom - top))
        return origin, radius

    def seed(self, robot_id, box, first, second, radius, speed, t_seconds):
        """Learn a port only from a locally observed outward launch, never a goal hit."""
        dx, dy = second[0] - first[0], second[1] - first[1]
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            return None
        direction = (dx / length, dy / length)
        left, top, right, bottom = box
        # Intersect the backwards ray with the source box to locate the exit edge.
        candidates = []
        for axis, bounds in ((0, (left, right)), (1, (top, bottom))):
            if abs(direction[axis]) <= 1e-6:
                continue
            for bound in bounds:
                distance = (second[axis] - bound) / direction[axis]
                p = (second[0] - direction[0] * distance,
                     second[1] - direction[1] * distance)
                if (distance >= 0 and left - 1 <= p[0] <= right + 1
                        and top - 1 <= p[1] <= bottom + 1):
                    candidates.append((distance, p))
        if not candidates:
            return None
        origin = min(candidates)[1]
        for port in self.ports:
            if port.robot_id != robot_id:
                continue
            old_origin, old_radius = self.geometry(port, box)
            if (sum(a * b for a, b in zip(direction, port.direction)) >= 0.94
                    and math.dist(origin, old_origin) <= 3 * max(radius, old_radius)):
                # Keep sampling geometry fixed during a burst: moving the strip would
                # itself create occupancy pulses. Refresh evidence, not the gate position.
                port.last_evidence = t_seconds
                return port
        scale = max(1.0, math.hypot(right - left, bottom - top))
        port = _Port(
            self.next_port_id, robot_id,
            ((origin[0] - left) / max(1.0, right - left),
             (origin[1] - top) / max(1.0, bottom - top)),
            direction, radius / scale, speed, t_seconds,
            ProminentPulseDetector(self.config.pulse_minimum_prominence,
                                  self.config.pulse_relative_prominence),
            ProminentPulseDetector(self.config.pulse_minimum_prominence,
                                  self.config.pulse_relative_prominence),
        )
        self.next_port_id += 1
        self.ports.append(port)
        # A bad input cannot grow an unbounded number of shooter hypotheses.
        self.ports[:] = self.ports[-24:]
        return port

    def register_track(self, port, box, points, shot, sample_interval):
        """Register interpolated inner-gate crossing time for event-level fusion."""
        origin, radius = self.geometry(port, box)
        target = radius
        distances = [sum((p.center[i] - origin[i]) * port.direction[i] for i in (0, 1))
                     for p in points]
        pair = next(((i - 1, i) for i in range(1, len(points))
                     if distances[i - 1] <= target <= distances[i]), None)
        if pair is None:
            # A ball first visible just beyond the strip still has a measured velocity.
            i = 0
            crossing_time = points[i].t_seconds - (distances[i] - target) / max(port.speed, 1)
            center = points[i].center
        else:
            a, b = pair
            fraction = (target - distances[a]) / max(1e-6, distances[b] - distances[a])
            crossing_time = points[a].t_seconds + fraction * (
                points[b].t_seconds - points[a].t_seconds)
            center = tuple(points[a].center[i] + fraction * (
                points[b].center[i] - points[a].center[i]) for i in (0, 1))
        tangent = (-port.direction[1], port.direction[0])
        lane = sum((center[i] - origin[i]) * tangent[i] for i in (0, 1))
        tolerance = max(sample_interval * 0.8, radius / max(port.speed, 1) * 0.65)
        for timestamp, old_lane, old_tolerance, existing in port.crossings:
            if (abs(timestamp - crossing_time) <= max(tolerance, old_tolerance)
                    and abs(lane - old_lane) < radius * 1.5):
                return existing
        port.crossings.append((crossing_time, lane, tolerance, shot))
        return None

    def register_pulse(self, pulse: GatePulse, box, shot, sample_interval):
        port = next(p for p in self.ports if p.port_id == pulse.port_id)
        origin, radius = self.geometry(port, box)
        tangent = (-port.direction[1], port.direction[0])
        lane = sum((pulse.inner.center[i] - origin[i]) * tangent[i] for i in (0, 1))
        tolerance = max(sample_interval * 0.8, radius / max(pulse.speed, 1) * 0.65)
        for timestamp, old_lane, old_tolerance, existing in port.crossings:
            if (abs(timestamp - pulse.inner.t_seconds) <= max(tolerance, old_tolerance)
                    and abs(lane - old_lane) < radius * 1.5):
                return existing
        port.crossings.append((pulse.inner.t_seconds, lane, tolerance, shot))
        return None

    def update(self, mask, robots, frame_index, t_seconds, minimum_speed) -> list[GatePulse]:
        import cv2

        self.last_samples = []
        self.ports[:] = [p for p in self.ports
                        if t_seconds - p.last_evidence <= self.config.gate_lifetime_seconds]
        events = []
        for port in self.ports:
            if port.robot_id not in robots:
                # No bridging across invisible/reidentified robot boxes.
                port.pending.clear()
                port.last_sample_time = None
                continue
            box = robots[port.robot_id].observation.bbox
            origin, radius = self.geometry(port, box)
            if port.previous_box is not None:
                old = port.previous_box
                change = max(abs(box[i] - old[i]) for i in range(4))
                if change > max(4 * radius, (box[2] - box[0]) * 0.3):
                    port.last_sample_time = None
            if (port.last_sample_time is None or t_seconds - port.last_sample_time
                    > self.config.maximum_observation_gap_seconds):
                port.inner = ProminentPulseDetector(self.config.pulse_minimum_prominence,
                                                   self.config.pulse_relative_prominence)
                port.outer = ProminentPulseDetector(self.config.pulse_minimum_prominence,
                                                   self.config.pulse_relative_prominence)
                port.pending.clear()
            interval = t_seconds - port.last_sample_time if port.last_sample_time is not None else 0
            port.previous_box = box
            port.last_sample_time = t_seconds
            tangent = (-port.direction[1], port.direction[0])
            peaks = []
            for gate_index, detector in enumerate((port.inner, port.outer)):
                distance = radius * (1 + gate_index * self.config.gate_separation_radii)
                center = tuple(origin[i] + port.direction[i] * distance for i in (0, 1))
                # Native-resolution strips: retain even a ball only a few pixels wide.
                u, v = np.meshgrid(np.arange(-0.4 * radius, 0.4 * radius + 1),
                                   np.arange(-2 * radius, 2 * radius + 1))
                x = (center[0] + u * port.direction[0] + v * tangent[0]).astype(np.float32)
                y = (center[1] + u * port.direction[1] + v * tangent[1]).astype(np.float32)
                if x.min() < 0 or y.min() < 0 or x.max() >= mask.shape[1] or y.max() >= mask.shape[0]:
                    peaks.append(None)
                    continue
                pixels = cv2.remap(mask, x, y, cv2.INTER_LINEAR).astype(float) / 255.0
                mass = float(pixels.sum())
                value = mass / max(1.0, 2 * radius * pixels.shape[1])
                observed_center = (float((x * pixels).sum() / mass),
                                   float((y * pixels).sum() / mass)) if mass > 0 else center
                sample = GateSample(frame_index, t_seconds, value, observed_center, radius)
                peaks.append(detector.update(sample))
                self.last_samples.append(dict(robot_id=port.robot_id, port_id=port.port_id,
                                              gate=gate_index, value=value, center=center,
                                              radius=radius, direction=port.direction))
            while port.pending and t_seconds - port.pending[0].t_seconds > (
                    self.config.maximum_transit_seconds + 2 * max(interval, 0)):
                port.pending.popleft()
            if peaks[0] is not None:
                port.pending.append(peaks[0])
            outer = peaks[1]
            if outer is None:
                continue
            possible = []
            for inner in port.pending:
                elapsed = outer.t_seconds - inner.t_seconds
                if not 0 < elapsed <= self.config.maximum_transit_seconds:
                    continue
                speed = radius * self.config.gate_separation_radii / elapsed
                if not max(minimum_speed, port.speed * 0.4) <= speed <= port.speed * 2.5:
                    continue
                lane_change = abs(sum((outer.center[i] - inner.center[i]) * tangent[i]
                                      for i in (0, 1)))
                if lane_change > radius * 1.5:
                    continue
                possible.append((abs(math.log(speed / max(port.speed, 1))), inner, speed))
            if not possible:
                continue
            _, inner, speed = min(possible, key=lambda item: item[0])
            port.pending.remove(inner)
            port.last_evidence = t_seconds
            events.append(GatePulse(port.robot_id, port.port_id, inner, outer, speed))
        return events
