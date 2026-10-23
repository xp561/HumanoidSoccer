"""Shared physics-step scoring for explicit cross-platform soccer comparisons."""
from dataclasses import dataclass, field
import math
import numpy as np


def spatial(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name}")
    return result


def goal_crossed(previous, current, goal, normal, width):
    """Directed XY goal-plane crossing, matching the existing G1 predicate."""
    previous = spatial(previous, 3, "previous ball position")
    current = spatial(current, 3, "current ball position")
    goal = spatial(goal, 3, "goal position")
    normal = spatial(normal, 2, "goal normal")
    length = np.linalg.norm(normal)
    if length <= 1e-12 or not math.isfinite(width) or width <= 0:
        raise ValueError("Invalid goal plane")
    normal = normal / length
    before = float(np.dot(previous[:2] - goal[:2], normal))
    after = float(np.dot(current[:2] - goal[:2], normal))
    if before > 0 or after < 0:
        return False
    denominator = before - after
    fraction = 1. if abs(denominator) < 1e-9 else before / denominator
    fraction = float(np.clip(fraction, 0., 1.))
    crossing = previous[:2] + fraction * (current[:2] - previous[:2])
    tangent = np.array([-normal[1], normal[0]])
    return bool(abs(float(np.dot(crossing - goal[:2], tangent))) <= width / 2)


@dataclass
class ShotMetrics:
    """Keep goal, kick, tracking and numerical outcomes separately inspectable.

    Buffer a control transition until its tracking flags are known. Mark a
    failure before submitting that transition's physics samples, so samples
    from an already-failed transition cannot enter the tracking-valid score.
    The fixed-horizon goal score continues after a tracking failure.
    """
    goal: np.ndarray
    normal: np.ndarray
    width: float
    outgoing_speed_threshold: float
    raw_goal: bool = False
    kicked_goal: bool = False
    tracking_valid_kicked_goal: bool = False
    contacted: bool = False
    first_contact_time: float | None = None
    impact_position: np.ndarray | None = None
    first_outgoing_cosine: float | None = None
    peak_outgoing_cosine: float | None = None
    max_post_contact_speed: float = 0.
    tracking_failure_time: float | None = None
    numerical_failure: bool = False
    last_time: float = field(default=-math.inf, repr=False)

    def __post_init__(self):
        self.goal = spatial(self.goal, 3, "goal").copy()
        self.normal = spatial(self.normal, 2, "normal").copy()
        length = np.linalg.norm(self.normal)
        if length <= 1e-12 or not math.isfinite(self.width) or self.width <= 0:
            raise ValueError("Invalid goal plane")
        self.normal /= length
        if not math.isfinite(self.outgoing_speed_threshold) or self.outgoing_speed_threshold <= 0:
            raise ValueError("Invalid outgoing speed threshold")

    def mark_tracking_failure(self, time):
        if not math.isfinite(time) or time < 0 or time < self.last_time:
            raise ValueError("Invalid failure time")
        if self.tracking_failure_time is None:
            self.tracking_failure_time = float(time)

    def observe(self, previous, current, velocity, foot_contact, time):
        if self.numerical_failure:
            raise RuntimeError("Cannot score a numerically invalid trajectory")
        previous = spatial(previous, 3, "previous ball position")
        current = spatial(current, 3, "current ball position")
        velocity = spatial(velocity, 3, "ball velocity")
        if not math.isfinite(time) or time <= self.last_time or time < 0:
            raise ValueError("Physics sample times must increase")
        self.last_time = float(time)
        if foot_contact and not self.contacted:
            self.contacted = True
            self.first_contact_time = float(time)
            self.impact_position = previous.copy()
        crossed = goal_crossed(previous, current, self.goal, self.normal, self.width)
        self.raw_goal |= crossed
        self.kicked_goal |= crossed and self.contacted
        self.tracking_valid_kicked_goal |= crossed and self.contacted and self.tracking_failure_time is None
        if not self.contacted:
            return
        speed = float(np.linalg.norm(velocity[:2]))
        direction = self.goal[:2] - self.impact_position[:2]
        distance = float(np.linalg.norm(direction))
        cosine = None
        if speed >= self.outgoing_speed_threshold and distance > 1e-12:
            cosine = float(np.clip(np.dot(velocity[:2], direction) / (speed * distance), -1., 1.))
            if self.first_outgoing_cosine is None:
                self.first_outgoing_cosine = cosine
        if speed > self.max_post_contact_speed:
            self.max_post_contact_speed = speed
            self.peak_outgoing_cosine = cosine

    def result(self):
        valid = not self.numerical_failure
        return {"goal": bool(self.raw_goal and valid),
                "kicked_goal": bool(self.kicked_goal and valid),
                "tracking_valid_kicked_goal": bool(self.tracking_valid_kicked_goal and valid),
                "foot_contact": self.contacted,
                "first_contact_time": self.first_contact_time,
                "impact_position": self.impact_position.tolist() if self.impact_position is not None else None,
                "kick_accuracy_cosine": self.first_outgoing_cosine if valid else None,
                "peak_speed_accuracy_cosine": self.peak_outgoing_cosine if valid else None,
                "max_post_contact_speed": self.max_post_contact_speed,
                "tracking_failed": self.tracking_failure_time is not None,
                "tracking_failure_time": self.tracking_failure_time,
                "numerical_failure": self.numerical_failure}
