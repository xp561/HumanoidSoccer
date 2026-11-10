"""Paired force-driven recovery; only fresh-world initialization restores state."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .balance import BalancePlanner
from .match_worker import ArenaPlant
from .match_arena import sha256
from .physics_worker import serial
from .readiness import Readiness, ReadinessConfig
from .walking_runtime import WalkingRuntime


def trial(snapshot, variant, output):
    receipt = json.loads(snapshot.with_suffix(".json").read_text())
    assert sha256(snapshot) == receipt["snapshot_sha256"]
    p = ArenaPlant(
        dict(motion=0, walking=True, arena_manifest=receipt["arena_manifest"])
    )
    r = p.robot
    with np.load(snapshot) as s:
        mujoco.mj_setState(r.model, r.data, s["integration_state"], int(s["spec"]))
        for name in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange"):
            getattr(r.model, name)[:] = s[name]
        mujoco.mj_forward(r.model, r.data)
        np.testing.assert_array_equal(r.data.qpos, s["qpos"])
        np.testing.assert_array_equal(r.data.qvel, s["qvel"])
        planner = BalancePlanner(p)
        nominal = p.motion["joint_pos"][0, p.meta["action_joint_ids"]].copy()
        if variant.startswith("old"):
            for name in ("positions", "rotations", "height", "orientation"):
                setattr(planner, name, s[name].copy())
    ids = np.asarray(p.meta["action_joint_ids"])
    terminal = r.data.qpos[r.qpos_ids[ids]].copy()
    ending = p.motion["joint_pos"][-1, ids].copy()
    all_feet = list(planner.feet)
    all_bounds = np.asarray(planner.bounds).copy()
    last_support = None
    c = ReadinessConfig()
    height = float(p.motion["body_pos_w"][0, p.root_id, 2])
    monitor = Readiness(c, height)
    posture = nominal.copy() if variant.endswith("_nominal") else terminal.copy()
    if variant == "pd_end":
        posture = ending.copy()
    loaded_count = 0
    landing_captured = False
    recovered = False
    unsafe = False
    start = float(r.data.time)
    rows = []
    recovery_s = None
    walker = None
    origin = None
    for step in range(600):
        elapsed = step * 0.02
        if recovered or variant == "walker_immediate":
            if walker is None:
                walker = WalkingRuntime()
            state = p.state()
            target = walker.target(
                joint_names=p.meta["joint_names"],
                joint_pos=state["joint_pos"],
                joint_vel=state["joint_vel"],
                root_quat=state["root_quat"],
                root_ang_vel=state["root_ang_vel"],
                command=[0.15, 0, 0] if recovered else [0, 0, 0],
            )
            request = dict(
                op="step",
                joint_names=list(walker.names),
                positions=target,
                kp=walker.kp,
                kd=walker.kd,
                max_torques=walker.effort_limit,
            )
        else:
            if variant.endswith("blend"):
                posture += np.clip(
                    nominal - posture, -c.joint_rate * 0.02, c.joint_rate * 0.02
                )
            if variant.startswith("landing"):
                posture += np.clip(ending - posture, -1.5 * 0.02, 1.5 * 0.02)
                diag = p.diagnostics()
                loaded_count = loaded_count + 1 if diag["both_feet_loaded"] else 0
                if loaded_count >= 3 and not landing_captured:
                    planner.capture()
                    planner.orientation = planner.upright(p.state()["root_quat"])
                    planner.height = height
                    landing_captured = True
                target = planner.target(posture) if landing_captured else posture.copy()
            elif variant.startswith("support"):
                diag = p.diagnostics()
                weight = float(r.model.body_subtreemass[r.root_id] * 9.81)
                support = tuple(
                    i
                    for i, force in enumerate(diag["foot_normal_forces"])
                    if force >= 0.02 * weight
                )
                if support and support != last_support:
                    planner.feet = [all_feet[i] for i in support]
                    planner.bounds = all_bounds[list(support)]
                    planner.capture()
                    planner.orientation = planner.upright(p.state()["root_quat"])
                    planner.height = height
                last_support = support
                target = planner.target(posture) if support else posture.copy()
            elif variant.startswith("pd"):
                target = posture.copy()
            else:
                target = planner.target(posture)
            request = dict(
                op="step",
                joint_names=p.meta["action_joint_names"],
                positions=target,
                kp=np.asarray(p.meta["stiffness"])[ids],
                kd=np.asarray(p.meta["damping"])[ids],
                max_torques=np.asarray(p.meta["effort_limits"])[ids],
            )
        p.step(
            {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in request.items()
            }
        )
        d = p.diagnostics()
        rows.append(
            dict(
                seconds=float(r.data.time) - start,
                walking=recovered,
                diagnostics=d,
                root_pos=p.state()["root_pos"].tolist(),
                target=np.asarray(target).tolist(),
            )
        )
        unsafe = bool(
            d["body_ground_contact"]
            or d["tilt"] > c.abort_tilt
            or abs(d["height"] - height) > c.abort_height_error
        )
        if unsafe:
            break
        if not recovered and monitor.update(float(r.data.time) - start, d):
            recovered = True
            recovery_s = float(r.data.time) - start
            walker = WalkingRuntime()
            origin = p.state()["root_pos"].copy()
        if recovered and float(r.data.time) - start >= recovery_s + 4 - 0.000001:
            break
        if not recovered and elapsed >= c.recovery_timeout:
            break
    result = dict(
        variant=variant,
        recovered=recovered,
        unsafe=unsafe,
        recovery_seconds=recovery_s,
        elapsed_seconds=float(r.data.time) - start,
        healthy_rewalk=recovered
        and not unsafe
        and float(r.data.time) - start >= recovery_s + 4 - 0.000001,
        walker_displacement_m=None
        if origin is None
        else float(np.linalg.norm(p.state()["root_pos"][:2] - origin[:2])),
        final_diagnostics=d,
        snapshot_sha256=receipt["snapshot_sha256"],
        robot_state_writes_after_initialization=0,
        trace=rows,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / (variant + ".json")).write_text(
        json.dumps(result, default=serial, indent=2)
    )
    print(
        json.dumps({k: v for k, v in result.items() if k != "trace"}, default=serial),
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    variants = [
        "old_nominal",
        "old_terminal",
        "capture_nominal",
        "capture_terminal",
        "capture_blend",
        "old_blend",
        "walker_immediate",
        "pd_terminal",
        "pd_blend",
        "support_terminal",
        "support_blend",
        "pd_end",
        "landing_end",
    ]
    ap.add_argument("--variants", nargs="+", choices=variants, default=variants)
    args = ap.parse_args()
    for variant in args.variants:
        trial(args.snapshot, variant, args.output)


if __name__ == "__main__":
    main()
