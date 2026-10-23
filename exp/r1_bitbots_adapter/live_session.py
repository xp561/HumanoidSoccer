"""Isolated ROS/MuJoCo preparation, execution and measured recovery session."""

import argparse
import base64
import json
import queue
import signal
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from bitbots_blackboard.capsules.kick_capsule import KickCapsule
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter

from bitbots_msgs.msg import JointCommand

from .handoff import Phase
from .mujoco_bridge import (
    Bridge,
    PlantClient,
    command_payload,
    sensor_barrier,
    spin_until,
    state_arrays,
)
from .ownership_ros import OwnershipRouter
from .readiness import Readiness, ReadinessConfig
from .ros_node import KickNode, forward_rotate


class Session:
    def __init__(self, output, *, realtime=True, render=True, config=None, joint_offset=0.0):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.config = config or ReadinessConfig()
        self.realtime = realtime
        self.render_enabled = render
        self.joint_offset = joint_offset
        self.requests = queue.Queue(maxsize=4)
        self.cancel = threading.Event()
        self.shutdown = threading.Event()
        self.lock = threading.RLock()
        self.status = {"stage": "initializing", "busy": False}
        self.jpeg = None
        self.counter = 0
        self.render_at = 0.0
        self.last_report = None
        self.events = []
        self.phase_trace = []
        self.control_pending = False
        self._last_wall = time.monotonic()
        self.plant = None

    def setup(self, *, rolling=False):
        self.adapter = KickNode(parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.router = OwnershipRouter(parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.bridge = Bridge(self.adapter, command_topic="leased_motor_goals")
        self.executor = SingleThreadedExecutor()
        for node in (self.adapter, self.router, self.bridge):
            self.executor.add_node(node)
        self.publishers = {
            name: self.bridge.create_publisher(JointCommand, topic, 10)
            for name, topic in (
                ("hold", "hold_motor_goals"),
                ("prepare", "prepare_motor_goals"),
                ("recover", "recovery_motor_goals"),
                ("hcm", "hcm_motor_goals"),
            )
        }
        self.kick_commands = []
        self.bridge.create_subscription(JointCommand, "kick_motor_goals", self.kick_commands.append, 100)
        self.reset(rolling=rolling)

    def reset(self, *, rolling=False):
        if self.adapter.busy:
            raise RuntimeError("Wait for the Action to finish before reset")
        if self.plant is not None:
            self.plant.close()
        # Restart the world only on an explicit new-trial reset.
        self.plant = PlantClient(
            dict(
                motion=0,
                ball_velocity=[0.2, 0.0, 0.0] if rolling else [0.0, 0.0, 0.0],
                initial_joint_offset=self.joint_offset,
            ),
            self.output / "worker.log",
            worker_module="exp.r1_bitbots_adapter.live_worker",
        )
        self.state = state_arrays(self.plant.initial)
        self.measured = self.plant.initial["diagnostics"]
        self.ns = max(getattr(self, "ns", 0), 1_000_000)
        self.nominal = self.adapter.references[0].data["joint_pos"][0, self.adapter.runtime.ids].copy()
        self.target = self.state["joint_pos"][self.adapter.runtime.ids].copy()
        self.posture = self.target.copy()
        self.capture_balance = True
        self.nominal_height = float(
            self.adapter.references[0].data["body_pos_w"][0, self.adapter.references[0].root, 2]
        )
        from .handoff import Handoff

        self.router.gate = Handoff(
            phase_timeout=max(
                self.config.preparation_timeout, self.config.recovery_timeout, self.adapter.cfg["execution_seconds"]
            )
            + 2.0
        )
        self.kick_commands.clear()
        self.router.forwarded.clear()
        self.router.rejected.clear()
        self.cancel.clear()
        self.goal = None
        self.metrics = None
        self.terminal_kick_count = None
        self.stage = "idle"
        self.reason = "new_trial; preparation not yet measured"
        self.outcome = None
        self.monitor = None
        self._last_wall = time.monotonic()
        self.sync(owner=False)
        self.update_view(force=True)

    @property
    def seconds(self):
        return self.ns / 1e9

    def sync(self, *, owner):
        self.bridge.sensors(self.state, self.ns, owner=owner)
        sensor_barrier(self.executor, self.adapter, self.ns, owner=owner)
        self.bridge.clock(self.ns)
        spin_until(
            self.executor, lambda: all(n.get_clock().now().nanoseconds == self.ns for n in (self.adapter, self.router))
        )

    def update_view(self, *, force=False):
        now = time.monotonic()
        if self.render_enabled and (force or now - self.render_at >= 0.12):
            jpeg = base64.b64decode(self.plant.rpc(dict(op="render"))["jpeg"])
            with self.lock:
                self.jpeg = jpeg
            self.render_at = time.monotonic()
        with self.lock:
            self.status = dict(
                stage=self.stage,
                busy=self.control_pending or self.stage in ("stopping", "preparing", "kicking", "recovering"),
                owner=self.router.gate.owner,
                reason=self.reason,
                outcome=self.outcome,
                simulation_seconds=round(self.seconds, 3),
                diagnostics=self.measured,
                readiness_reasons=[] if self.monitor is None else self.monitor.reasons,
                stable_seconds=0.0 if self.monitor is None else round(self.monitor.elapsed, 3),
                required_stable_seconds=self.config.dwell,
                direction_body=[1.0, 0.0],
                goal_world=self.goal,
                ball_world=self.state["ball_pos"].tolist(),
                last_report=self.last_report,
                realtime_requested=self.realtime,
                config=asdict(self.config),
                scope="simulation only; READY grants hold ownership, not walking",
            )

    def consume(self, command):
        reply = self.plant.rpc(command_payload(command))
        self.state = state_arrays(reply)
        self.measured = reply["diagnostics"]
        self.metrics = reply.get("metrics", self.metrics)
        self.target = np.array(command.positions, float)
        self.ns += round(self.adapter.runtime.dt * 1e9)
        self.router.gate.tick(self.ns)
        if self.stage in ("stopping", "preparing", "kicking", "recovering"):
            self.phase_trace.append(
                dict(
                    seconds=self.seconds,
                    stage=self.stage,
                    diagnostics=self.measured,
                    root_pos=self.state["root_pos"].tolist(),
                    root_quat=self.state["root_quat"].tolist(),
                    joint_pos=self.state["joint_pos"].tolist(),
                    ball_pos=self.state["ball_pos"].tolist(),
                    target=self.target.tolist(),
                )
            )
        self.update_view()
        if self.realtime:
            delay = self.adapter.runtime.dt - (time.monotonic() - self._last_wall)
            if delay > 0:
                self.shutdown.wait(delay)
            self._last_wall = time.monotonic()
        return reply

    def controlled_step(self, source, target):
        if source != "hcm":
            planned = self.plant.rpc(dict(op="plan", posture=np.asarray(target).tolist(), capture=self.capture_balance))
            self.capture_balance = False
            target = planned["positions"]
        self.sync(owner=False)
        before = len(self.bridge.commands)
        command = JointCommand()
        command.header.stamp = self.bridge.stamp(self.ns)
        rt = self.adapter.runtime
        command.joint_names = list(rt.action_names)
        command.positions = np.asarray(target, float).tolist()
        command.kp, command.kd = rt.kp.tolist(), rt.kd.tolist()
        command.max_torques = rt.effort.tolist()
        self.publishers[source].publish(command)
        spin_until(self.executor, lambda: len(self.bridge.commands) > before)
        if len(self.bridge.commands) != before + 1:
            raise RuntimeError("Unexpected competing motor command")
        return self.consume(self.bridge.commands[-1])

    def unsafe(self):
        m = self.measured
        return (
            m["body_ground_contact"]
            or abs(self.nominal_height - m["height"]) > self.config.abort_height_error
            or m["tilt"] > self.config.abort_tilt
        )

    def event(self, stage, reason):
        self.stage, self.reason = stage, reason
        self.events.append(dict(stage=stage, reason=reason, seconds=self.seconds))
        self.update_view()

    def wait_ready(self, *, preparation, timeout, interruptible=False):
        self.monitor = Readiness(self.config, self.nominal_height)
        start = self.seconds
        while self.seconds - start < timeout and not self.shutdown.is_set():
            if self.unsafe():
                return "unsafe_posture"
            if (preparation or interruptible) and self.cancel.is_set():
                return "cancelled"
            if self.router.gate.phase == Phase.FAULT:
                return self.router.gate.reason
            if preparation:
                self.posture += np.clip(
                    self.nominal - self.posture,
                    -self.config.joint_rate * self.adapter.runtime.dt,
                    self.config.joint_rate * self.adapter.runtime.dt,
                )
            self.controlled_step(self.router.gate.owner, self.posture.copy())
            if self.terminal_kick_count is not None and len(self.kick_commands) != self.terminal_kick_count:
                return "policy_command_after_terminal_result"
            error = float(np.max(np.abs(self.state["joint_pos"][self.adapter.runtime.ids] - self.nominal)))
            if self.monitor.update(self.seconds, self.measured, pose_error=error if preparation else None):
                return None
        return (
            "stopping_timeout"
            if self.stage == "stopping"
            else "preparation_timeout"
            if preparation
            else "recovery_timeout"
        )

    def shot(self):
        self.cancel.clear()
        self.events, self.phase_trace = [], []
        self.shot_kick_start = len(self.kick_commands)
        self.terminal_kick_count = None
        self.outcome = None
        generation = self.router.gate.begin(self.ns)
        self.target = self.state["joint_pos"][self.adapter.runtime.ids].copy()
        self.posture = self.target.copy()
        self.capture_balance = True
        self.event("stopping", "waiting for measured standstill")
        failure = self.wait_ready(preparation=False, timeout=self.config.preparation_timeout, interruptible=True)
        if failure is None and not self.cancel.is_set():
            self.router.gate.acknowledge(Phase.STOPPING, generation, self.ns)
            self.event("preparing", "converging to the policy startup pose")
            failure = self.wait_ready(preparation=True, timeout=self.config.preparation_timeout)
        elif self.cancel.is_set():
            failure = "cancelled"
        if failure is not None:
            self.outcome = failure
            if failure != "cancelled":
                self.fault(failure)
                self.save_report(False)
                return
            self.router.gate.fail("cancelled_before_execution", self.ns)
            generation = self.router.gate.recover_fault(self.ns)
        else:
            self.router.gate.acknowledge(Phase.PREPARING, generation, self.ns)
            self.event("kicking", "fixed-policy Action; completion does not imply recovery")
            failure = self.execute_kick()
            self.outcome = failure or "policy_completed"
            if self.router.gate.phase == Phase.FAULT:
                self.fault(self.router.gate.reason)
                self.save_report(False)
                return
            self.router.gate.acknowledge(Phase.KICKING, generation, self.ns, outcome=self.outcome)
        self.target = self.state["joint_pos"][self.adapter.runtime.ids].copy()
        self.posture = self.target.copy() if self.outcome == "policy_completed" else self.nominal.copy()
        self.capture_balance = self.outcome == "policy_completed"
        self.plant.rpc(dict(op="end_policy"))
        self.event("recovering", "balance feedback from terminal pose; checking sustained support")
        failure = self.wait_ready(preparation=False, timeout=self.config.recovery_timeout)
        if failure is not None:
            self.fault(failure)
            self.save_report(False)
            return
        self.router.gate.acknowledge(Phase.RECOVERING, generation, self.ns)
        self.event("ready", "measured stable recovery; hold owner only")
        self.save_report(True)

    def execute_kick(self):
        self.sync(owner=True)
        q = self.state["root_quat"]
        forward = forward_rotate(q, [1.0, 0.0, 0.0])
        yaw = float(np.arctan2(forward[1], forward[0]))
        board = SimpleNamespace(
            config={"rl_kick": {"strength_mode": "fixed", "timeout": -1.0}},
            world_model=SimpleNamespace(get_current_position=lambda: (0.0, 0.0, yaw)),
        )
        capsule = KickCapsule(self.bridge, board)
        try:
            # Validate geometry before publishing any policy target.
            from bitbots_msgs.action import Kick

            request = Kick.Goal(x=1.0, y=0.0, strength=0.0, timeout=-1.0)
            _, goal, _ = self.adapter._request(request)
            self.goal = goal.tolist()
            self.plant.rpc(dict(op="goal", goal=self.goal))
            before = len(self.bridge.commands)
            if not capsule.start_rl_kick(float(np.degrees(yaw))):
                return capsule.last_rl_kick_error or "request_failed"
            spin_until(
                self.executor, lambda: capsule._rl_kick_goal_handle is not None or not capsule.is_currently_kicking
            )
            handle = capsule._rl_kick_goal_handle
            if handle is None:
                return self.adapter.last_failure or "request_rejected"
            future = handle.get_result_async()
            while not self.shutdown.is_set():
                if self.cancel.is_set() or self.unsafe():
                    capsule.stop_rl_kick()
                spin_until(self.executor, lambda before=before: len(self.bridge.commands) > before or future.done())
                if future.done():
                    break
                if len(self.bridge.commands) != before + 1:
                    raise RuntimeError("Duplicate policy command")
                command = self.bridge.commands[-1]
                before += 1
                if command.header.stamp != self.bridge.stamp(self.ns):
                    raise RuntimeError("Policy command is not synchronized to physics")
                self.consume(command)
                self.sync(owner=True)
            if not future.done():
                capsule.stop_rl_kick()
                spin_until(self.executor, future.done)
            result = future.result()
            spin_until(self.executor, lambda: not self.adapter.busy)
            for _ in range(3):
                self.executor.spin_once(timeout_sec=0.001)
            self.terminal_kick_count = len(self.kick_commands)
            if self.unsafe():
                self.router.gate.fail("unsafe_posture", self.ns)
            if result.status == GoalStatus.STATUS_SUCCEEDED:
                return None
            return (
                "cancelled"
                if result.status == GoalStatus.STATUS_CANCELED
                else self.adapter.last_failure or "action_aborted"
            )
        except ValueError as error:
            return str(error)
        finally:
            capsule._rl_kick_client.destroy()

    def fault(self, reason):
        if self.router.gate.phase != Phase.FAULT:
            self.router.gate.fail(reason, self.ns)
        self.target = self.state["joint_pos"][self.adapter.runtime.ids].copy()
        self.posture = self.target.copy()
        self.capture_balance = True
        self.event("fault", reason)

    def save_report(self, recovered):
        self.counter += 1
        path = self.output / f"trial_{self.counter:04d}.json"
        report = dict(
            recovered=recovered,
            shot_metrics=self.metrics,
            policy_commands=len(self.kick_commands) - self.shot_kick_start,
            policy_commands_after_terminal=0
            if self.terminal_kick_count is None
            else len(self.kick_commands) - self.terminal_kick_count,
            outcome=self.outcome,
            final_stage=self.stage,
            final_owner=self.router.gate.owner,
            final_diagnostics=self.measured,
            readiness_config=asdict(self.config),
            events=self.events,
            ownership_transitions=self.router.gate.transitions,
            source="measured MuJoCo states and external contact forces",
            physics_dt=self.plant.initial["physics_dt"],
            policy_dt=self.adapter.runtime.dt,
            reset_state_writes_after_trial_start=0,
            scope="simulation preparation/recovery; no walking or get-up controller",
            trace=self.phase_trace,
        )
        path.write_text(json.dumps(report, indent=2))
        self.last_report = str(path)
        self.update_view()

    def run(self):
        try:
            self.setup()
            while not self.shutdown.is_set():
                try:
                    request = self.requests.get_nowait()
                except queue.Empty:
                    request = None
                if request and request["op"] == "reset":
                    self.reset(rolling=request.get("rolling", False))
                elif request and request["op"] == "start" and self.stage in ("idle", "ready"):
                    self.shot()
                else:
                    if self.stage in ("idle", "ready") and self.unsafe():
                        self.fault("unsafe_posture_while_holding")
                    self.controlled_step(self.router.gate.owner, self.posture)
                if request is not None:
                    self.control_pending = False
                # Long-running viewing must not retain an unbounded ROS command history.
                if self.stage != "kicking":
                    for history in (self.bridge.commands, self.router.forwarded, self.router.rejected):
                        if len(history) > 512:
                            del history[:-128]
        except Exception as error:
            with self.lock:
                self.status.update(stage="error", reason=str(error), busy=False)
            raise
        finally:
            if self.plant is not None:
                self.plant.close()
            if hasattr(self, "executor"):
                self.executor.shutdown(timeout_sec=3)
                for node in (self.bridge, self.router, self.adapter):
                    node.destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("logs/r1_logic_fix/r1_live"))
    parser.add_argument("--port", type=int, default=8877)
    args = parser.parse_args()
    from .live_web import serve

    rclpy.init()

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    session = Session(args.output)
    try:
        serve(session, args.port)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
