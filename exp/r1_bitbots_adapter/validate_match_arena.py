"""Compile, inspect and physically exercise generated arena geometry."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from .live_worker import LivePlant
from .match_arena import field_to_world, sha256, world_to_field
from .match_geometry import ArenaTracker, FieldGeometry
from .match_worker import ArenaPlant
from .walking_runtime import WalkingRuntime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidate = ArenaPlant(dict(motion=0, arena_manifest=str(args.candidate)))
    reference = ArenaPlant(
        dict(motion=0, walking=True, arena_manifest=str(args.reference))
    )
    original = LivePlant(dict(motion=0, walking=True))
    a, b = original.robot.model, reference.robot.model
    assert (a.nq, a.nv, a.nu) == (b.nq, b.nv, b.nu)
    for name in (
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "jnt_qposadr",
        "jnt_dofadr",
        "dof_armature",
    ):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for name in original.meta["body_names"]:
        ai, bi = a.body(name).id, b.body(name).id
        for attr in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
            np.testing.assert_array_equal(getattr(a, attr)[ai], getattr(b, attr)[bi])
    ground_a, ground_b = a.geom("ground").id, b.geom("ground").id
    np.testing.assert_array_equal(a.geom_friction[ground_a], b.geom_friction[ground_b])
    manifest = candidate.manifest
    c = manifest["config"]
    model = candidate.robot.model
    for name in manifest["visual_only_geometries"]:
        geom = model.geom(name).id
        assert model.geom_contype[geom] == model.geom_conaffinity[geom] == 0
    for sign in (-1, 1):
        ids = [model.geom(f"arena_post_{sign}_{edge}").id for edge in (-1, 1)]
        centres = [model.geom_pos[i] for i in ids]
        inner = centres[1][1] - centres[0][1] - 2 * c["post_radius"]
        assert abs(inner - c["goal_clear_width"]) < 1e-10
        bar = model.geom(f"arena_bar_{sign}").id
        assert (
            abs(
                model.geom_pos[bar, 2]
                - model.geom_size[bar, 0]
                - c["goal_clear_height"]
            )
            < 1e-10
        )
    measured_ball = model.geom("soccer_ball_geom").id
    assert abs(model.geom_size[measured_ball, 0] - manifest["ball"]["radius"]) < 1e-12
    assert (
        abs(model.body_mass[model.body("soccer_ball").id] - manifest["ball"]["mass"])
        < 1e-12
    )
    walker = WalkingRuntime()
    maximum_difference = 0.0
    for i in range(150):
        state = original.state()
        target = walker.target(
            joint_names=original.meta["joint_names"],
            joint_pos=state["joint_pos"],
            joint_vel=state["joint_vel"],
            root_quat=state["root_quat"],
            root_ang_vel=state["root_ang_vel"],
            command=[0, 0, 0] if i < 50 else [0.15, 0, 0],
        )
        payload = dict(
            op="step",
            joint_names=walker.names,
            positions=target.tolist(),
            kp=walker.kp.tolist(),
            kd=walker.kd.tolist(),
            max_torques=walker.effort_limit.tolist(),
        )
        original.step(payload)
        reference.step(payload)
        diff = float(
            np.max(np.abs(original.robot.data.qpos - reference.robot.data.qpos))
        )
        maximum_difference = max(maximum_difference, diff)
    assert reference.arena_status()["physics_samples_observed"] == 3000
    assert maximum_difference < 1e-10, maximum_difference
    assert not reference.diagnostics()["body_ground_contact"]

    # Launch probes use fresh data, never reset a running autonomous session.
    cases = [
        ("positive_goal", [4.2, 0, 0.07], [2, 0, 0], "goal", "goal_line"),
        ("negative_goal", [-4.2, 0, 0.07], [-2, 0, 0], "goal", "goal_line"),
        ("sideline", [0, 2.8, 0.07], [0, 2, 0], "out", "touchline"),
        ("miss_wide", [4.2, 1.6, 0.07], [2, 0, 0], "out", "goal_line"),
        ("above_bar", [4.3, 0, 1.6], [4, 0, 0], "out", "goal_line"),
        ("post_collision", [4.15, 0.95, 0.07], [3, 0, 0], None, None),
    ]
    probes = []
    ball_geom = model.geom("soccer_ball_geom").id
    for label, pos, velocity, kind, boundary in cases:
        d = mujoco.MjData(model)
        d.qpos[:] = candidate.robot.data.qpos
        d.qpos[candidate.robot.ball_q : candidate.robot.ball_q + 3] = field_to_world(
            pos, candidate.heading
        )
        d.qvel[candidate.robot.ball_v : candidate.robot.ball_v + 3] = field_to_world(
            velocity, candidate.heading
        )
        d.ctrl[:] = 0
        # Keep unrelated robot motion out of ball launch geometry probes.
        d.qpos[candidate.robot.root_q : candidate.robot.root_q + 2] = [20, 20]
        tracker = ArenaTracker(FieldGeometry(c), 0.07, pos)
        post_contacts = 0
        for _ in range(600):
            mujoco.mj_step(model, d)
            point = world_to_field(
                d.qpos[candidate.robot.ball_q : candidate.robot.ball_q + 3],
                candidate.heading,
            )
            tracker.observe(point, float(d.time))
            for contact in d.contact:
                other = (
                    contact.geom2
                    if contact.geom1 == ball_geom
                    else contact.geom1
                    if contact.geom2 == ball_geom
                    else -1
                )
                if other >= 0 and model.geom(other).name.startswith("arena_post_"):
                    post_contacts += 1
        event = None if not tracker.events else tracker.events[0]
        if label == "post_collision":
            assert post_contacts > 0
            assert event is None or event["kind"] != "goal"
        else:
            assert event and event["kind"] == kind and event["boundary"] == boundary, (
                label,
                event,
            )
        probes.append(
            dict(
                case=label, passed=True, event=event, post_contact_samples=post_contacts
            )
        )

    model.vis.global_.offwidth = 1100
    model.vis.global_.offheight = 800
    renderer = mujoco.Renderer(model, height=800, width=1100)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0]
    camera.distance = 13
    camera.azimuth = np.degrees(candidate.heading) - 90
    camera.elevation = -89.5
    renderer.update_scene(candidate.robot.data, camera=camera)
    Image.fromarray(renderer.render()).save(args.output / "arena_mujoco_topdown.png")
    camera.distance = 7.5
    camera.azimuth = np.degrees(candidate.heading) + 220
    camera.elevation = -35
    renderer.update_scene(candidate.robot.data, camera=camera)
    Image.fromarray(renderer.render()).save(args.output / "arena_mujoco_oblique.png")
    renderer.close()
    receipt = dict(
        passed=True,
        geometry_probes=probes,
        robot_contract_unchanged=True,
        visual_lines_noncolliding=True,
        frozen_ball_dynamics_max_qpos_difference=maximum_difference,
        frozen_ball_regression_seconds=3.0,
        frozen_model_sha256=sha256(manifest["source_model"]),
        robot_model_mass_kg=float(model.body_subtreemass[candidate.robot.root_id]),
        measured_goal_clear_width_m=c["goal_clear_width"],
        measured_goal_clear_height_m=c["goal_clear_height"],
        physics_dt=float(model.opt.timestep),
        candidate_ball=manifest["ball"],
        scope="geometry and contact validation; fresh launch data per probe; not a policy success-rate test",
    )
    (args.output / "result.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
