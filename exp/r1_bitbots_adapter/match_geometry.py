"""Small-field geometry in metres; no ROS, physics or policy dependencies.

The origin is field centre; goals face along +/-x. Dimensions describe the
outside edges of painted boundaries. Goal width/height are CLEAR openings.
The goal plane is the goal-side edge of the goal line, at x=+/-length/2.
"""

import json
import math
from pathlib import Path


# MuJoCo soft ground contacts can place the sphere a few micrometres below z=r.
# This applies only to the ground floor, never to side lines, posts or the bar.
GROUND_CONTACT_TOLERANCE = 0.001

DEFAULT_CONFIG = Path(__file__).with_name("arena") / "small_2026.json"


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    if config["schema_version"] != 1:
        raise ValueError("Unsupported arena schema")
    dimensions = (
        "field_length",
        "field_width",
        "padding",
        "line_width",
        "goal_clear_width",
        "goal_clear_height",
        "goal_depth",
        "post_radius",
        "goal_area_depth",
        "goal_area_width",
        "penalty_area_depth",
        "penalty_area_width",
        "penalty_mark_distance",
        "center_circle_diameter",
        "mark_diameter",
    )
    for key in dimensions:
        value = config[key]
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Invalid dimension: {key}")
    if not (
        config["goal_clear_width"]
        < config["goal_area_width"]
        < config["penalty_area_width"]
        < config["field_width"]
    ):
        raise ValueError("Invalid nested area widths")
    if not (
        config["line_width"]
        < config["goal_area_depth"]
        < config["penalty_area_depth"]
        < config["field_length"] / 2
    ):
        raise ValueError("Invalid nested area depths")
    for ball in config["ball_profiles"].values():
        if not all(math.isfinite(ball[k]) and ball[k] > 0 for k in ("radius", "mass")):
            raise ValueError("Invalid ball profile")
    return config


class FieldGeometry:
    def __init__(self, config=None):
        self.config = load_config() if config is None else config
        self.half_length = self.config["field_length"] / 2
        self.half_width = self.config["field_width"] / 2

    @staticmethod
    def validate(position, radius):
        if len(position) != 3 or not all(math.isfinite(x) for x in position):
            raise ValueError("Ball position must contain three finite coordinates")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("Ball radius must be finite and positive")

    def overlaps_field(self, position, radius):
        """Horizontal ball projection intersects the closed field rectangle.

        This handles rounded corner exits correctly, unlike axis tests alone.
        Height does not affect whether an airborne ball is in play.
        """
        self.validate(position, radius)
        dx = max(abs(position[0]) - self.half_length, 0.0)
        dy = max(abs(position[1]) - self.half_width, 0.0)
        return dx * dx + dy * dy <= radius * radius

    def in_goal_volume(self, position, radius):
        """Static volume membership, not evidence of a legal goal entry."""
        self.validate(position, radius)
        x, y, z = position
        return (
            self.half_length + radius
            < abs(x)
            <= self.half_length + self.config["goal_depth"] - radius
            and abs(y) + radius <= self.config["goal_clear_width"] / 2
            and z >= radius - GROUND_CONTACT_TOLERANCE
            and z + radius <= self.config["goal_clear_height"]
        )

    def crossing(self, before, after, radius):
        """First complete exit during one straight sampled segment.

        Start outside => no new event. Interpolate the first exit to avoid
        missed fast crossings and distinguish a high ball from a valid goal.
        Call on PHYSICS substeps; not on sparse control/video samples.
        """
        self.validate(before, radius)
        self.validate(after, radius)
        if not self.overlaps_field(before, radius) or self.overlaps_field(
            after, radius
        ):
            return None
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            point = [a + mid * (b - a) for a, b in zip(before, after)]
            if self.overlaps_field(point, radius):
                lo = mid
            else:
                hi = mid
        t = (lo + hi) / 2
        x, y, z = [a + t * (b - a) for a, b in zip(before, after)]
        end = abs(abs(x) - self.half_length - radius) < 1e-9
        side = abs(abs(y) - self.half_width - radius) < 1e-9
        mouth = (
            abs(y) + radius <= self.config["goal_clear_width"] / 2
            and z >= radius - GROUND_CONTACT_TOLERANCE
            and z + radius <= self.config["goal_clear_height"]
        )
        goal = end and mouth
        return dict(
            kind="goal" if goal else "out",
            boundary="goal_line" if end else "touchline" if side else "corner",
            end=1 if x >= 0 else -1,
            touchline_side=1 if y >= 0 else -1,
            segment_fraction=t,
            crossing_position=[x, y, z],
        )


class ArenaTracker:
    """Latch one terminal event per play; restart is external referee work."""

    def __init__(self, geometry, radius, initial_position):
        geometry.validate(initial_position, radius)
        self.geometry = geometry
        self.radius = radius
        self.previous = list(initial_position)
        self.events = []

    def observe(self, position, seconds):
        if not math.isfinite(seconds):
            raise ValueError("Nonfinite physics timestamp")
        self.geometry.validate(position, self.radius)
        event = (
            None
            if self.events
            else self.geometry.crossing(self.previous, position, self.radius)
        )
        if event is not None:
            event["sample_end_seconds"] = seconds
            self.events.append(event)
        self.previous = list(position)
        return event


def bitbots_parameters(config):
    """Adapt clear opening to Bit-Bots' documented post-centre distance."""
    return {
        "field.name": config["id"],
        "field.size.x": config["field_length"],
        "field.size.y": config["field_width"],
        "field.size.padding": config["padding"],
        "field.goal.width": config["goal_clear_width"] + 2 * config["post_radius"],
        "field.goal.depth": config["goal_depth"],
        "field.markings.penalty_area.size.x": config["penalty_area_depth"],
    }
