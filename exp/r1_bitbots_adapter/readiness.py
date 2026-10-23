"""Measured, sustained readiness for the isolated simulation controller."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReadinessConfig:
    dwell: float = 0.5
    max_sample_gap: float = 0.05
    height_error: float = 0.10
    tilt: float = 0.20
    linear_speed: float = 0.15
    angular_speed: float = 0.50
    joint_speed: float = 0.80
    pose_error: float = 0.20
    abort_tilt: float = 0.80
    abort_height_error: float = 0.25
    preparation_timeout: float = 8.0
    recovery_timeout: float = 8.0
    joint_rate: float = 0.35

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid simulation readiness setting: " + name)
        if self.abort_tilt <= self.tilt or self.abort_height_error <= self.height_error:
            raise ValueError("Abort envelope must contain the readiness envelope")


class Readiness:
    def __init__(self, config, nominal_height):
        self.config = config
        self.nominal_height = nominal_height
        self.since = None
        self.previous = None
        self.reasons = ["not_measured"]
        self.elapsed = 0.0

    def update(self, now, measured, *, pose_error=None):
        c = self.config
        values = [now, *[measured[k] for k in ("height", "tilt", "linear_speed", "angular_speed", "joint_speed")]]
        if pose_error is not None:
            values.append(pose_error)
        if not all(math.isfinite(x) for x in values):
            self.since = None
            self.reasons = ["nonfinite_measurement"]
            self.elapsed = 0.0
            return False
        if self.previous is not None and now <= self.previous:
            self.since = None
            self.elapsed = 0.0
            self.reasons = ["nonincreasing_measurement_time"]
            return False
        if self.previous is not None and now - self.previous > c.max_sample_gap:
            self.since = None
        self.previous = now
        checks = {
            "height": abs(measured["height"] - self.nominal_height) <= c.height_error,
            "tilt": measured["tilt"] <= c.tilt,
            "linear_speed": measured["linear_speed"] <= c.linear_speed,
            "angular_speed": measured["angular_speed"] <= c.angular_speed,
            "joint_speed": measured["joint_speed"] <= c.joint_speed,
            "both_feet_loaded": measured["both_feet_loaded"] is True,
            "body_ground_contact": measured["body_ground_contact"] is False,
        }
        if pose_error is not None:
            checks["preparation_pose"] = pose_error <= c.pose_error
        self.reasons = [name for name, ok in checks.items() if not ok]
        if self.reasons:
            self.since = None
            self.elapsed = 0.0
            return False
        if self.since is None:
            self.since = now
        self.elapsed = now - self.since
        return self.elapsed + 1e-9 >= c.dwell
