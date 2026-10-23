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


@dataclass
class KickTrial:
    success: bool = False
    contacted: bool = False
    max_ball_speed: float = 0.0
    kick_accuracy_cosine: float = math.nan

    def observe(
        self,
        previous_ball: np.ndarray,
        current_ball: np.ndarray,
        ball_velocity: np.ndarray,
        goal_center: np.ndarray,
        goal_normal: np.ndarray,
        goal_width: float,
        contact_speed: float,
    ) -> None:
        self.success |= goal_crossed(previous_ball, current_ball, goal_center, goal_normal, goal_width)
        speed = float(np.linalg.norm(ball_velocity[:2]))
        if speed < contact_speed:
            return
        self.contacted = True
        if speed <= self.max_ball_speed:
            return
        direction = goal_center[:2] - current_ball[:2]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 1.0e-9:
            self.kick_accuracy_cosine = float(np.clip(np.dot(ball_velocity[:2], direction) / (speed * direction_norm), -1.0, 1.0))
        self.max_ball_speed = speed
