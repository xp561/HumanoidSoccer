"""R1/Isaac Lab adapter for the shared soccer evaluation definitions.

The success predicate is intentionally imported from the existing G1 MuJoCo
evaluator so R1 and G1 reports use the same goal-plane definition.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.mujoco_soccer.metrics import goal_crossed


def goal_plane_normal(goal_center: np.ndarray, environment_origin: np.ndarray) -> np.ndarray:
    """Keep the goal plane fixed when the initial ball position changes."""
    direction = goal_center[..., :2] - environment_origin[..., :2]
    length = np.linalg.norm(direction, axis=-1, keepdims=True)
    if np.any(length < 1e-9):
        raise ValueError("Goal center must differ from the environment origin")
    return direction / length


@dataclass
class KickTrial:
    success: bool = False
    contacted: bool = False
    max_ball_speed: float = 0.0
    kick_accuracy_cosine: float = math.nan
    finished: bool = False
    end_reason: str = "horizon"

    def finish(self, reason: str) -> None:
        if not self.finished:
            self.finished = True
            self.end_reason = reason

    def observe(
        self,
        previous_ball: np.ndarray,
        current_ball: np.ndarray,
        ball_velocity: np.ndarray,
        goal_center: np.ndarray,
        goal_normal: np.ndarray,
        goal_width: float,
        contact_speed: float,
        foot_contact: bool,
    ) -> None:
        if self.finished:
            return
        self.contacted |= foot_contact
        self.success |= self.contacted and goal_crossed(
            previous_ball, current_ball, goal_center, goal_normal, goal_width
        )
        speed = float(np.linalg.norm(ball_velocity[:2]))
        self.max_ball_speed = max(self.max_ball_speed, speed)
        if not self.contacted or speed < contact_speed or math.isfinite(self.kick_accuracy_cosine):
            return
        direction = goal_center[:2] - current_ball[:2]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 1.0e-9:
            self.kick_accuracy_cosine = float(np.clip(np.dot(ball_velocity[:2], direction) / (speed * direction_norm), -1.0, 1.0))
