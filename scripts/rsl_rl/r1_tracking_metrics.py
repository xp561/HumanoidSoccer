"""Tracking metrics for Isaac Lab R1 rollouts.

The position and velocity metrics intentionally match the existing MuJoCo
``eval_baseline_metrics`` definitions.  AUJ is reported separately as a
transparent peak-link-jerk deviation proxy, since the original repository did
not implement the paper's AUJ aggregation.
"""
from __future__ import annotations

import math

import numpy as np


def _mean_norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(values, axis=-1).mean()) if values.size else math.nan


def _finite_difference(values: np.ndarray) -> np.ndarray:
    return values[1:] - values[:-1]


def _peak_link_jerk(positions: np.ndarray) -> np.ndarray:
    """Per-link peak absolute third difference, averaged over xyz components."""
    if positions.shape[0] < 4:
        return np.full(positions.shape[1], np.nan)
    jerk = np.diff(positions, n=3, axis=0)
    return np.abs(jerk).sum(axis=-1).max(axis=0)


def evaluate_tracking(
    policy_body_pos: np.ndarray,
    reference_body_pos: np.ndarray,
    policy_joint_pos: np.ndarray,
    reference_joint_pos: np.ndarray,
) -> dict[str, float]:
    """Evaluate one rollout using the existing repository's tracking formulas."""
    if policy_body_pos.shape != reference_body_pos.shape:
        raise ValueError(f"Body trajectory shapes differ: {policy_body_pos.shape} vs {reference_body_pos.shape}")
    if policy_joint_pos.shape != reference_joint_pos.shape:
        raise ValueError(f"Joint trajectory shapes differ: {policy_joint_pos.shape} vs {reference_joint_pos.shape}")
    pol_zero = policy_body_pos - policy_body_pos[0, 0]
    ref_zero = reference_body_pos - reference_body_pos[0, 0]
    global_error = np.linalg.norm(pol_zero - ref_zero, axis=-1).mean(axis=-1)
    rel_policy = pol_zero - pol_zero[:, :1]
    rel_reference = ref_zero - ref_zero[:, :1]
    local_error = np.linalg.norm(rel_policy - rel_reference, axis=-1).mean(axis=-1)
    pol_velocity = _finite_difference(pol_zero)
    ref_velocity = _finite_difference(ref_zero)
    policy_peak = _peak_link_jerk(policy_body_pos)
    reference_peak = _peak_link_jerk(reference_body_pos)
    return {
        "frames": int(policy_body_pos.shape[0]),
        "E_gmpjpe_mm": float(global_error.mean() * 1000.0),
        "E_mpjpe_mm": float(local_error.mean() * 1000.0),
        "E_vel_mm_per_frame": _mean_norm(pol_velocity[:, :1] - ref_velocity[:, :1]) * 1000.0,
        "E_dof_mpjpe_mm": _mean_norm(policy_joint_pos - reference_joint_pos) * 1000.0,
        "AUJ_proxy_peak_link_jerk_deviation": float(np.abs(policy_peak - reference_peak).mean()),
        "policy_mean_peak_link_jerk": float(policy_peak.mean()),
        "reference_mean_peak_link_jerk": float(reference_peak.mean()),
    }


def tracking_trace(policy_body_pos: np.ndarray, reference_body_pos: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-frame diagnostics using the same root-origin alignment as G-MPJPE."""
    policy_zero = policy_body_pos - policy_body_pos[0, 0]
    reference_zero = reference_body_pos - reference_body_pos[0, 0]
    global_delta = policy_zero - reference_zero
    body_error_mm = np.linalg.norm(global_delta, axis=-1) * 1000.0
    local_delta = (policy_zero - policy_zero[:, :1]) - (reference_zero - reference_zero[:, :1])
    return {
        "global_mpjpe_mm": body_error_mm.mean(axis=1),
        "root_position_error_mm": body_error_mm[:, 0],
        "local_mpjpe_mm": np.linalg.norm(local_delta, axis=-1).mean(axis=1) * 1000.0,
        "per_body_error_mm": body_error_mm,
    }
