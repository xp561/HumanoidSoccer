"""Offline regressions for motion conversion, contact gating, and trial isolation."""

import ast
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))
sys.path.insert(0, str(ROOT))

from motion_archive import G1_JOINT_NAMES, R1_CONTROLLED_JOINT_NAMES, load_g1_motion, source_joint_indices, validate_joint_limits
from r1_eval_metrics import KickTrial, goal_plane_normal


def archive_fields():
    positions = np.tile(np.arange(len(G1_JOINT_NAMES)), (3, 1)).astype(float)
    return {
        "fps": np.array([50.]), "kick_leg": np.array("right"),
        "joint_pos": positions, "joint_vel": positions * 2,
        "body_pos_w": np.zeros((3, 30, 3)), "body_quat_w": np.zeros((3, 30, 4)),
        "body_lin_vel_w": np.zeros((3, 30, 3)), "body_ang_vel_w": np.zeros((3, 30, 3)),
    }


def test_exporter_order_maps_knees_and_preserves_leg(tmp_path):
    path = tmp_path / "right.npz"
    np.savez(path, **archive_fields())
    motion = load_g1_motion(path)
    indices = source_joint_indices(motion, ("left_knee_joint", "right_knee_joint"))
    np.testing.assert_array_equal(motion["joint_pos"][:, indices], [[9, 10]] * 3)
    np.testing.assert_array_equal(motion["joint_vel"][:, indices], [[18, 20]] * 3)
    assert motion["kick_leg"].item() == "right"


def test_explicit_metadata_overrides_exporter_order(tmp_path):
    fields = archive_fields()
    fields["joint_names"] = np.asarray(G1_JOINT_NAMES[::-1])
    fields["joint_pos"] = fields["joint_pos"][:, ::-1]
    fields["joint_vel"] = fields["joint_vel"][:, ::-1]
    path = tmp_path / "shuffled.npz"
    np.savez(path, **fields)
    motion = load_g1_motion(path)
    indices = source_joint_indices(motion, ("left_knee_joint", "right_knee_joint"))
    np.testing.assert_array_equal(motion["joint_pos"][:, indices], [[9, 10]] * 3)


@pytest.mark.parametrize("defect", ["missing_leg", "unknown_leg", "duplicate_names", "nan", "fps"])
def test_invalid_archive_is_rejected(tmp_path, defect):
    fields = archive_fields()
    if defect == "missing_leg":
        del fields["kick_leg"]
    elif defect == "unknown_leg":
        fields["kick_leg"] = np.array("unknown")
    elif defect == "duplicate_names":
        fields["joint_names"] = np.asarray([G1_JOINT_NAMES[0]] * len(G1_JOINT_NAMES))
    elif defect == "nan":
        fields["joint_pos"][0, 0] = np.nan
    else:
        fields["fps"] = np.array([0.])
    path = tmp_path / "invalid.npz"
    np.savez(path, **fields)
    with pytest.raises(ValueError):
        load_g1_motion(path)


def test_joint_limit_violation_is_not_silently_clipped():
    with pytest.raises(ValueError, match="ankle"):
        validate_joint_limits(np.array([[-.9]]), np.array([[-.26, .26]]), ("ankle",))


def observe(trial, previous, current, contact=True, velocity=(0., -2., 0.)):
    trial.observe(np.array(previous), np.array(current), np.array(velocity),
                  np.array([0., -5., .11]), np.array([0., -1.]), 1., .5, contact)


def test_forward_goal_and_reverse_rejection():
    forward, reverse = KickTrial(), KickTrial()
    observe(forward, [0, -4.9, .11], [0, -5.1, .11])
    observe(reverse, [0, -5.1, .11], [0, -4.9, .11])
    assert forward.success
    assert not reverse.success


def test_goal_normal_points_outward_and_is_translation_invariant():
    origin = np.array([[10., 20., 0.]])
    goal = np.array([[10., 15., .11]])
    np.testing.assert_array_equal(goal_plane_normal(goal, origin), [[0., -1.]])
    with pytest.raises(ValueError):
        goal_plane_normal(origin, origin)


def test_speed_alone_does_not_count_as_contact_or_goal():
    trial = KickTrial()
    observe(trial, [0, -4.9, .11], [0, -5.1, .11], contact=False)
    assert not trial.contacted
    assert not trial.success
    assert np.isnan(trial.kick_accuracy_cosine)


def test_reset_cannot_join_two_attempts():
    trial = KickTrial()
    observe(trial, [0, -1, .11], [0, -2, .11])
    trial.finish("episode_end")
    observe(trial, [0, -4.9, .11], [0, -5.1, .11])
    assert not trial.success
    assert trial.end_reason == "episode_end"


def test_accuracy_keeps_first_post_contact_sample():
    trial = KickTrial()
    observe(trial, [0, -1, .11], [0, -2, .11], velocity=(2., 0., 0.))
    first_accuracy = trial.kick_accuracy_cosine
    observe(trial, [0, -2, .11], [0, -3, .11], velocity=(0., -6., 0.))
    assert trial.kick_accuracy_cosine == first_accuracy
    assert trial.max_ball_speed == 6.


@pytest.fixture
def contact_module():
    # Execute the production contact logic with only the simulator's type imports removed.
    path = ROOT / "source/whole_body_tracking/soccer/tasks/tracking/mdp/kick_detection.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("isaaclab")
    )]
    module = types.ModuleType("offline_contact_logic")
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    yield module
    del sys.modules[module.__name__]


@pytest.mark.parametrize("history, expected", [
    ([[-20, 0, 0], [0, 0, 0], [0, 0, 0]], 20.),
    ([[8, 0, 0], [0, 8, 0], [0, 0, 8]], 8.),
    ([[0, 0, 30], [0, 0, 0]], 0.),
])
def test_horizontal_peak_handles_negative_and_vertical_forces(contact_module, history, expected):
    forces = torch.tensor([history], dtype=torch.float32).unsqueeze(2)
    assert contact_module.peak_horizontal_force(forces).item() == expected


def test_pair_contact_selects_actual_foot_and_retains_unknown_label(contact_module):
    forces = torch.zeros(1, 3, 1, 2, 3)
    forces[0, 0, 0, 1, 0] = -20.
    sensor = SimpleNamespace(data=SimpleNamespace(force_matrix_w_history=forces))
    positions = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
    robot = SimpleNamespace(data=SimpleNamespace(body_pos_w=positions))
    class Scene:
        sensors = {"ball": sensor}
        env_origins = torch.zeros(1, 3)
        def __getitem__(self, name):
            return robot
    env = SimpleNamespace(device="cpu", num_envs=1, scene=Scene())
    tracker = contact_module.KickContactTracker(env, "_motion")
    tracker._foot_cache = (torch.tensor([0, 1]), torch.tensor([0, 1], dtype=torch.int8))
    command = SimpleNamespace(robot=robot, soccer_ball_pos=torch.zeros(1, 3), kick_leg=torch.tensor([-1]))
    event = tracker.detect(command, "ball", 10.)
    assert event.new_contact.item()
    foot = tracker.resolve_contact_foot(command, SimpleNamespace(name="robot"), event.new_contact)
    assert foot.sides.item() == 1
    assert foot.expected.item() == -1
    tracker.begin_step(command)
    assert not tracker.detect(command, "ball", 10.).new_contact.item()


def test_released_motions_fit_r1_limits_after_correct_mapping():
    import xml.etree.ElementTree as ET
    directory = ROOT / "motions/soccer-standard"
    if not directory.is_dir():
        pytest.skip("Released motion data not present")
    urdf = ET.parse(ROOT / "source/whole_body_tracking/soccer/assets/unitree_description/urdf/r1/main.urdf")
    limits = {joint.get("name"): [float(joint.find("limit").get(key)) for key in ("lower", "upper")]
              for joint in urdf.findall("joint") if joint.find("limit") is not None
              and joint.find("limit").get("lower") is not None}
    for path in sorted(directory.glob("*.npz")):
        motion = load_g1_motion(path)
        indices = source_joint_indices(motion, R1_CONTROLLED_JOINT_NAMES)
        validate_joint_limits(motion["joint_pos"][:, indices],
                              np.asarray([limits[name] for name in R1_CONTROLLED_JOINT_NAMES]),
                              R1_CONTROLLED_JOINT_NAMES)


def test_regenerated_archives_preserve_named_joint_states_and_labels():
    directory = ROOT / "motions/r1-corrected"
    if not directory.is_dir():
        pytest.skip("Regenerate the corrected R1 references first")
    sources = sorted((ROOT / "motions/soccer-standard").glob("*.npz"))
    assert sources
    assert {path.name for path in sources} == {path.name for path in directory.glob("*.npz")}
    for path in sources:
        source = load_g1_motion(path)
        with np.load(directory / path.name, allow_pickle=False) as target:
            names = target["joint_names"].tolist()
            source_indices = source_joint_indices(source, R1_CONTROLLED_JOINT_NAMES)
            target_indices = [names.index(name) for name in R1_CONTROLLED_JOINT_NAMES]
            np.testing.assert_allclose(target["joint_pos"][:, target_indices], source["joint_pos"][:, source_indices])
            np.testing.assert_allclose(target["joint_vel"][:, target_indices], source["joint_vel"][:, source_indices])
            assert target["kick_leg"].item() == source["kick_leg"].item()
            np.testing.assert_array_equal(target["fps"], source["fps"])
            for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                assert np.isfinite(target[key]).all()
            np.testing.assert_allclose(np.linalg.norm(target["body_quat_w"], axis=-1), 1., atol=1e-5)


def test_motion_selection_does_not_mix_robot_and_environment_frames():
    path = ROOT / "source/whole_body_tracking/soccer/tasks/tracking/mdp/commands_multi_motion_soccer.py"
    tree = ast.parse(path.read_text())
    command_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MotionCommand")
    method = next(node for node in command_class.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_select_motion_for_eval_ball")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    command = SimpleNamespace(
        _to_env_id_tensor=lambda ids: torch.as_tensor(ids, dtype=torch.long),
        soccer_ball_pos=torch.tensor([[0., -1., .11], [1., 0., .11]]),
        motion_nominal_ball_xy=torch.tensor([[0., -1.], [1., 0.]]),
        motion_local_displacement_xy=torch.tensor([[1., 0.], [0., -1.]]),
        motion_idx=torch.zeros(2, dtype=torch.long),
        motion_length=torch.zeros(2, dtype=torch.long),
        time_steps=torch.ones(2, dtype=torch.long),
        motion=SimpleNamespace(file_lengths=torch.tensor([10, 20])),
    )
    namespace["_select_motion_for_eval_ball"](command, [0, 1])
    assert command.motion_idx.tolist() == [0, 1]
    assert command.motion_length.tolist() == [10, 20]
    assert command.time_steps.tolist() == [0, 0]
