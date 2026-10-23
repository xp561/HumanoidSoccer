"""Lockstep ROS/MuJoCo integration with an isolated original-version plant."""

import argparse
import json
import os
import select
import subprocess
import time
from pathlib import Path

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from bitbots_msgs.action import Kick
from bitbots_msgs.msg import JointCommand, RobotControlState
from exp.r1_sim2sim.interface import inverse_rotate

from .ros_node import KickNode, forward_rotate
from .runtime import Runtime

ROOT = Path.cwd()
FORMAL = ROOT / "logs/r1_logic_fix/r1_mujoco_formal_20261017"
STATE_KEYS = ("root_pos", "root_quat", "root_ang_vel", "joint_pos", "joint_vel", "ball_pos")


class PlantClient:
    def __init__(self, case, log, *, worker_module="exp.r1_bitbots_adapter.physics_worker"):
        env = dict(os.environ)
        for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
            env.pop(name, None)
        env["PYTHONNOUSERSITE"] = "1"
        self.log = open(log, "w")
        self.process = subprocess.Popen(
            ["/home/tl/miniconda3/envs/isaaclab/bin/python", "-u", "-m", worker_module],
            cwd=ROOT,
            env=env,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            bufsize=1,
        )
        self.initial = self.rpc(dict(op="init", **case))

    def rpc(self, message):
        if self.process.poll() is not None:
            raise RuntimeError("Physics worker exited")
        self.process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], 30)
        if not ready:
            raise TimeoutError("Physics worker response timed out")
        reply = json.loads(self.process.stdout.readline())
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def close(self):
        try:
            if self.process.poll() is None:
                self.rpc(dict(op="close"))
                self.process.wait(timeout=5)
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.log.close()


class Bridge(Node):
    def __init__(self, adapter, command_topic="kick_motor_goals"):
        super().__init__(
            "r1_mujoco_bridge",
            namespace="/r1_adapter_test",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.adapter = adapter
        self.commands = []
        self.feedback = []
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self.odom_pub = self.create_publisher(Odometry, "pelvis_odometry", 10)
        self.ball_pub = self.create_publisher(PoseWithCovarianceStamped, "ball_position_relative_filtered", 10)
        self.state_pub = self.create_publisher(RobotControlState, "robot_state", 10)
        self.owner_pub = self.create_publisher(Bool, "walking_released", 10)
        self.create_subscription(JointCommand, command_topic, self.commands.append, 100)
        self.client = ActionClient(self, Kick, "rl_kick")

    @staticmethod
    def stamp(ns):
        return rclpy.time.Time(nanoseconds=int(ns)).to_msg()

    def clock(self, ns):
        msg = Clock()
        msg.clock = self.stamp(ns)
        self.clock_pub.publish(msg)

    def sensors(self, state, ns, *, ball=True, owner=True):
        stamp = self.stamp(ns)
        joint = JointState()
        joint.header.stamp = stamp
        joint.name = list(self.adapter.runtime.names[::-1])
        joint.position = [float(x) for x in state["joint_pos"][::-1]]
        joint.velocity = [float(x) for x in state["joint_vel"][::-1]]
        self.joint_pub.publish(joint)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "pelvis_link"
        root = state["root_pos"]
        quat = state["root_quat"]
        p, q = odom.pose.pose.position, odom.pose.pose.orientation
        p.x, p.y, p.z = map(float, root)
        q.w, q.x, q.y, q.z = map(float, quat)
        angular = inverse_rotate(quat, state["root_ang_vel"])
        a = odom.twist.twist.angular
        a.x, a.y, a.z = map(float, angular)
        self.odom_pub.publish(odom)
        if ball:
            msg = PoseWithCovarianceStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = "odom"
            p = msg.pose.pose.position
            p.x, p.y, p.z = map(float, state["ball_pos"])
            self.ball_pub.publish(msg)
        self.state_pub.publish(RobotControlState(state=RobotControlState.CONTROLLABLE))
        self.owner_pub.publish(Bool(data=owner))


def spin_until(executor, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("ROS lockstep barrier timed out")
        executor.spin_once(timeout_sec=0.002)


def sensor_barrier(executor, adapter, ns, *, ball=True, owner=True):
    expected = ns / 1e9

    def ready():
        return (
            adapter.joints is not None
            and adapter.odom is not None
            and abs(adapter.joints[0] - expected) < 1e-9
            and abs(adapter.stamp(adapter.odom) - expected) < 1e-9
            and (not ball or (adapter.ball is not None and abs(adapter.ball[0] - expected) < 1e-9))
            and adapter.owner is not None
            and adapter.owner[1] == owner
        )

    spin_until(executor, ready)


def command_payload(command):
    return dict(
        op="step",
        joint_names=list(command.joint_names),
        positions=list(command.positions),
        kp=list(command.kp),
        kd=list(command.kd),
        max_torques=list(command.max_torques),
    )


def target_payload(target):
    return dict(
        op="step",
        joint_names=list(target.joint_names),
        positions=target.positions.tolist(),
        kp=target.kp.tolist(),
        kd=target.kd.tolist(),
        max_torques=target.effort_limits.tolist(),
    )


def state_arrays(reply):
    return {name: np.asarray(value, float) for name, value in reply["state"].items()}


def save_trace(path, trace):
    np.savez(path, **{name: np.asarray(value) for name, value in trace.items()})


def append_trace(trace, state, obs, action, target):
    for name, value in {**state, "obs": obs, "action": action, "target": target}.items():
        trace.setdefault(name, []).append(np.array(value, copy=True))


def direct(case, ref, goal, output):
    runtime = Runtime(FORMAL / "artifacts")
    plant = PlantClient(case, output.with_suffix(".worker.log"))
    trace = {}
    state = state_arrays(plant.initial)
    plant.rpc(dict(op="goal", goal=np.asarray(goal).tolist()))
    final = None
    try:
        for step in range(round(8 / runtime.dt)):
            obs = runtime.observation(
                **{k: state[k] for k in STATE_KEYS}, **ref.sample(step * runtime.dt), goal_pos=goal
            )
            action = runtime.step(obs)
            target = runtime.targets(action)
            append_trace(trace, state, obs, action, target.positions)
            final = plant.rpc(target_payload(target))
            state = state_arrays(final)
    finally:
        plant.close()
    save_trace(output.with_suffix(".npz"), trace)
    output.write_text(json.dumps(final, indent=2))
    return final


def ros_trial(case, output, fault=None, ownership=False, bitbots_fixed=False):
    rclpy.init()
    adapter = KickNode(parameter_overrides=[Parameter("use_sim_time", value=True)])
    router=None
    if ownership:
        from .ownership_ros import OwnershipRouter
        router=OwnershipRouter(parameter_overrides=[Parameter("use_sim_time",value=True)])
    bridge = Bridge(adapter,command_topic="leased_motor_goals" if ownership else "kick_motor_goals")
    interference=bridge.create_publisher(JointCommand,"walking_motor_goals",100) if ownership else None
    executor = SingleThreadedExecutor()
    executor.add_node(adapter)
    executor.add_node(bridge)
    if router is not None:
        executor.add_node(router)
    plant = PlantClient(case, output.with_suffix(".worker.log"))
    dt_ns = round(adapter.runtime.dt * 1e9)
    epoch = 1_000_000
    trace = {}
    policy_inputs = []
    original_step = adapter.runtime.step

    def traced_step(obs):
        policy_inputs.append(np.array(obs, copy=True))
        return original_step(obs)

    adapter.runtime.step = traced_step
    state = state_arrays(plant.initial)
    final = None
    fault_at = 50
    last_command_count = 0
    try:
        bridge.clock(epoch)
        spin_until(executor, lambda: adapter.get_clock().now().nanoseconds == epoch)
        bridge.sensors(state, epoch)
        sensor_barrier(executor, adapter, epoch)
        spin_until(executor, lambda: adapter.state is not None and bridge.client.server_is_ready())
        request = Kick.Goal()
        request.x, request.y, request.strength, request.timeout = 1.0, 0.0, 0.0, -1.0
        ref, goal, _ = adapter._request(request)
        selected = adapter.references.index(ref)
        if selected != case["motion"]:
            raise RuntimeError(f"Reference selection mismatch: expected {case['motion']}, got {selected}")
        plant.rpc(dict(op="goal", goal=goal.tolist()))
        transaction=None
        if router is not None:
            from .handoff import Phase
            spin_until(executor,lambda:router.get_clock().now().nanoseconds==epoch)
            transaction=router.begin_transaction()
            router.acknowledge_phase(Phase.STOPPING,transaction)
            router.acknowledge_phase(Phase.PREPARING,transaction)
        if bitbots_fixed:
            from types import SimpleNamespace
            from bitbots_blackboard.capsules.kick_capsule import KickCapsule

            q = state["root_quat"]
            forward = forward_rotate(q, [1, 0, 0])
            yaw = float(np.arctan2(forward[1], forward[0]))
            board = SimpleNamespace(
                config={"rl_kick": {"strength_mode": "fixed", "timeout": -1.0}},
                world_model=SimpleNamespace(get_current_position=lambda: (0.0, 0.0, yaw)),
            )
            capsule = KickCapsule(bridge, board)
            if not capsule.start_rl_kick(float(np.degrees(yaw))):
                raise RuntimeError("Bit-Bots fixed-policy request failed")
            spin_until(executor, lambda: capsule._rl_kick_goal_handle is not None or not capsule.is_currently_kicking)
            handle = capsule._rl_kick_goal_handle
            if handle is None:
                raise RuntimeError("Bit-Bots request rejected: " + str(adapter.last_failure))
        else:
            sent = bridge.client.send_goal_async(
                request, feedback_callback=lambda msg: bridge.feedback.append(msg.feedback.time_remaining)
            )
            spin_until(executor, sent.done)
            handle = sent.result()
        if not handle.accepted:
            raise RuntimeError("Kick rejected: " + str(adapter.last_failure))
        result_future = handle.get_result_async()
        full_steps = round(adapter.cfg["execution_seconds"] / adapter.runtime.dt)
        for step in range(full_steps + 1):
            ns = epoch + step * dt_ns
            if interference is not None and step < full_steps:
                bogus=JointCommand()
                bogus.header.stamp=bridge.stamp(ns)
                bogus.joint_names=list(adapter.runtime.action_names)
                bogus.positions=(adapter.runtime.offset+.4).tolist()
                bogus.kp=adapter.runtime.kp.tolist()
                bogus.kd=adapter.runtime.kd.tolist()
                bogus.max_torques=adapter.runtime.effort.tolist()
                bogus.from_hcm=True
                interference.publish(bogus)
            if step:
                if fault == "cancel" and step == fault_at:
                    cancelled = handle.cancel_goal_async()
                    spin_until(executor, cancelled.done)
                    if not cancelled.result().goals_canceling:
                        raise RuntimeError("Cancellation was not accepted")
                ball = fault != "ball_dropout" or step < fault_at
                owner = fault != "owner_loss" or step < fault_at
                bridge.sensors(state, ns, ball=ball, owner=owner)
                sensor_barrier(executor, adapter, ns, ball=ball, owner=owner)
                bridge.clock(ns)
            spin_until(executor, lambda count=last_command_count: len(bridge.commands) > count or result_future.done())
            if result_future.done():
                break
            if len(bridge.commands) != last_command_count + 1:
                raise RuntimeError("Duplicate or missed control command")
            command = bridge.commands[-1]
            command_ns = command.header.stamp.sec * 1_000_000_000 + command.header.stamp.nanosec
            if command_ns != ns:
                raise RuntimeError(f"Control timestamp mismatch at frame {step}: {command_ns} != {ns}")
            if len(policy_inputs) != last_command_count + 1:
                raise RuntimeError("Inference/command count mismatch")
            expected_obs = policy_inputs[-1]
            action = adapter.runtime.last_action.copy()
            append_trace(trace, state, expected_obs, action, command.positions)
            last_command_count += 1
            final = plant.rpc(command_payload(command))
            state = state_arrays(final)
        if not result_future.done():
            spin_until(executor, result_future.done)
        result = result_future.result()
        count_at_end = len(bridge.commands)
        if router is not None:
            router.acknowledge_phase(Phase.KICKING,transaction,outcome=str(result.status))
        # Simulator-only ownership fallback: stop the policy stream and hold current joint positions.
        # This is not an HCM walking or recovery controller.
        for extra in range(1, 11):
            final_hold = plant.rpc(dict(op="hold"))
            state = state_arrays(final_hold)
            ns = epoch + (step + extra) * dt_ns
            bridge.sensors(state, ns)
            sensor_barrier(executor, adapter, ns)
            bridge.clock(ns)
            spin_until(executor, lambda expected=ns: adapter.get_clock().now().nanoseconds == expected)
            for _ in range(3):
                executor.spin_once(timeout_sec=0.001)
        if len(bridge.commands) != count_at_end:
            raise RuntimeError("Policy command published after terminal result")
        expected_status = (
            GoalStatus.STATUS_SUCCEEDED
            if fault is None
            else GoalStatus.STATUS_CANCELED
            if fault == "cancel"
            else GoalStatus.STATUS_ABORTED
        )
        if result.status != expected_status:
            raise RuntimeError(f"Unexpected terminal state {result.status}; {adapter.last_failure}")
        if fault is None and last_command_count != full_steps:
            raise RuntimeError("Full interval did not receive exactly one command per control frame")
        report = dict(
            case=case,
            fault=fault,
            action_status=result.status,
            result_code=result.result.result,
            policy_commands=last_command_count,
            post_terminal_commands=len(bridge.commands) - count_at_end,
            controlled_seconds=last_command_count * adapter.runtime.dt,
            end_clock=(ns - epoch) / 1e9,
            goal=goal.tolist(),
            reference=adapter.runtime.meta["motion_names"][selected],
            physical_mujoco_version=plant.initial["mujoco"],
            physics_dt=plant.initial["physics_dt"],
            first_control_frame=0,
            final_controlled=final,
            simulator_hold=final_hold,
            failure=adapter.last_failure,
            passed=True,
            ownership=None if router is None else dict(
                forwarded=len(router.forwarded),rejected=len(router.rejected),
                walking_forwarded=sum(x["source"]=="walk" for x in router.forwarded),
                kick_forwarded=sum(x["source"]=="kick" for x in router.forwarded),
                final_phase=router.gate.phase.value,transitions=router.gate.transitions,
                readiness_source="test acknowledgements; initial pose checked by policy server"),
        )
        if router is not None and (report["ownership"]["walking_forwarded"] != 0
                                  or report["ownership"]["kick_forwarded"] != last_command_count):
            raise RuntimeError("Ownership routing violation")
        save_trace(output.with_suffix(".npz"), trace)
        output.write_text(json.dumps(report, indent=2))
        return report, ref, goal
    finally:
        plant.close()
        executor.shutdown(timeout_sec=3)
        bridge.destroy_node()
        if router is not None:
            router.destroy_node()
        adapter.destroy_node()
        rclpy.shutdown()


def compare_pair(ros_path, direct_path):
    ros = np.load(ros_path)
    baseline = np.load(direct_path)
    errors = {}
    for name in ros.files:
        if ros[name].shape != baseline[name].shape:
            raise RuntimeError("Trajectory shape mismatch: " + name)
        errors[name] = float(np.max(np.abs(ros[name] - baseline[name])))
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=int, default=0)
    parser.add_argument("--rolling", action="store_true")
    parser.add_argument("--ownership-router",action="store_true")
    parser.add_argument("--bitbots-fixed", action="store_true")
    parser.add_argument("--fault", choices=["cancel", "ball_dropout", "owner_loss"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    case = dict(motion=args.motion, ball_velocity=[0.2, 0, 0] if args.rolling else [0, 0, 0])
    ros_path = args.output / "ros.json"
    ros, ref, goal = ros_trial(case, ros_path, args.fault,args.ownership_router,args.bitbots_fixed)
    result = dict(ros=ros, scope="ROS/physics integration diagnostic, not a formal success-rate estimate.")
    if args.fault is None:
        baseline_path = args.output / "direct.json"
        baseline = direct(case, ref, goal, baseline_path)
        errors = compare_pair(ros_path.with_suffix(".npz"), baseline_path.with_suffix(".npz"))
        result["max_abs_trajectory_error"] = errors
        result["baseline"] = baseline
        score_keys = ("goal", "kicked_goal", "tracking_valid_kicked_goal", "foot_contact")
        result["score_agreement"] = all(
            ros["final_controlled"]["metrics"][k] == baseline["metrics"][k] for k in score_keys
        )
        result["passed"] = max(errors.values()) < 2e-5 and result["score_agreement"]
    else:
        result["passed"] = ros["passed"]
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise RuntimeError("ROS/MuJoCo paired comparison failed")


if __name__ == "__main__":
    main()
