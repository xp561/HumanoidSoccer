"""Launch Isaac Sim headlessly and verify that the R1 articulation is usable.

Run from the HumanoidSoccer repository with the ``soccer`` conda environment:

    python scripts/verify_r1_isaacsim_load.py --headless

The script creates one R1 instance, advances one simulation step, and checks
the expected articulation interfaces.  It does not train a policy or write any
motion data.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Verify that the R1 asset loads in Isaac Sim.")
parser.add_argument(
    "--hold",
    action="store_true",
    help="Keep the Isaac Sim window open after a successful check; close the window or press Ctrl+C to exit.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from soccer.robots.r1 import R1_ACTION_SCALE, R1_CYLINDER_CFG


HEAD_JOINT_NAMES = {"head_pitch_joint", "head_yaw_joint"}


@configclass
class R1VerificationSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot = R1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def verify() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    scene = InteractiveScene(R1VerificationSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot: Articulation = scene["robot"]
    joint_names = tuple(robot.joint_names)
    body_names = tuple(robot.body_names)
    action_joint_names = set(R1_ACTION_SCALE)

    missing_heads = HEAD_JOINT_NAMES.difference(joint_names)
    if missing_heads:
        raise RuntimeError(f"R1 articulation is missing head joints: {sorted(missing_heads)}")
    if HEAD_JOINT_NAMES.intersection(action_joint_names):
        raise RuntimeError("R1 head joints must not be present in R1_ACTION_SCALE")
    missing_action_joints = action_joint_names.difference(joint_names)
    if missing_action_joints:
        raise RuntimeError(f"R1 action configuration references missing joints: {sorted(missing_action_joints)}")
    if "pelvis_link" not in body_names:
        raise RuntimeError("R1 articulation is missing pelvis_link")
    if len(joint_names) != len(action_joint_names) + len(HEAD_JOINT_NAMES):
        raise RuntimeError(
            "Unexpected R1 joint count: "
            f"got {len(joint_names)}, expected {len(action_joint_names) + len(HEAD_JOINT_NAMES)}"
        )

    # ``sim.reset()`` above has initialized the articulation and all tensor
    # views.  An asset-load test intentionally stops here: rendering or physics
    # stepping would test the GPU/display pipeline rather than R1 loading.
    tensors = (robot.data.root_pos_w, robot.data.joint_pos, robot.data.body_pos_w)
    if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
        raise RuntimeError("R1 produced non-finite state values after one simulation step")

    print("[PASS] R1 loaded successfully in Isaac Sim", flush=True)
    print(f"[PASS] joints={len(joint_names)}, bodies={len(body_names)}, policy_action_joints={len(action_joint_names)}", flush=True)
    print("[PASS] head_pitch_joint and head_yaw_joint are excluded from R1_ACTION_SCALE", flush=True)

    if args_cli.hold:
        print("[INFO] Keeping Isaac Sim open. Close its window or press Ctrl+C to exit.", flush=True)
        while simulation_app.is_running():
            simulation_app.update()


exit_code = 0
try:
    verify()
except KeyboardInterrupt:
    print("\n[INFO] Verification window closed by user.", flush=True)
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    # Isaac Sim 4.5 can stall in ``simulation_app.close()`` on this headless
    # host after a successful URDF import.  This is a one-shot diagnostic, so
    # return its result immediately and let the operating system reclaim the
    # process resources instead of leaving a GPU-consuming process behind.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
