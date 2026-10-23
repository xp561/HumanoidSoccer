"""Isolated ROS transport prototype for offline/simulation integration only."""

import math
import os
import threading
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener

from bitbots_msgs.action import Kick
from bitbots_msgs.msg import JointCommand, RobotControlState
from exp.r1_sim2sim.interface import inverse_rotate, vector

from .runtime import Reference, Runtime, anchored_goal, fresh_stamp


def quaternion(msg):
    return vector([msg.w, msg.x, msg.y, msg.z], 4, "ROS orientation")


def forward_rotate(q, value):
    q = vector(q, 4, "orientation")
    return inverse_rotate(q * [1, -1, -1, -1], value)


class KickNode(Node):
    def __init__(self, **kwargs):
        super().__init__("r1_kick_adapter", namespace="/r1_adapter_test", **kwargs)
        if not self.get_namespace().startswith("/r1_adapter_test/") and self.get_namespace() != "/r1_adapter_test":
            raise RuntimeError("Prototype is restricted to the isolated simulation namespace")
        if not os.environ.get("ROS_DOMAIN_ID"):
            raise RuntimeError("Set an isolated ROS_DOMAIN_ID for this prototype")
        default_bundle = str(Path.cwd() / "logs/r1_logic_fix/r1_mujoco_formal_20261017/artifacts")
        parameters = {
            "bundle": default_bundle,
            "motions": str(Path.cwd() / "motions/r1-corrected"),
            "sensor_max_age": 0.15,
            "state_max_age": 0.5,
            "execution_seconds": 8.0,
            "goal_distance": 5.0,
            "goal_height": 0.11,
            "start_joint_error": 0.2,
            "start_joint_speed": 10.0,
            "start_yaw_error": 0.05,
            "start_tilt_error": 0.1,
            "max_ball_reference_distance": 0.5,
            "heading_tolerance": 0.01,
        }
        for name, value in parameters.items():
            self.declare_parameter(name, value)
        self.cfg = {name: self.get_parameter(name).value for name in parameters}
        for name in parameters:
            if name not in ("bundle", "motions"):
                value = self.cfg[name]
                if not np.isfinite(value) or value <= 0:
                    raise ValueError("Invalid parameter " + name)
        self.runtime = Runtime(self.cfg["bundle"])
        self.references = [
            Reference(Path(self.cfg["motions"]) / (name + ".npz"), self.runtime)
            for name in self.runtime.meta["motion_names"]
        ]
        self.nominal_body = []
        for index, ref in enumerate(self.references):
            root = ref.data["body_pos_w"][0, ref.root]
            ball = np.array([*self.runtime.meta["motion_nominal_ball_xy"][index], self.cfg["goal_height"]])
            self.nominal_body.append(inverse_rotate(ref.data["body_quat_w"][0, ref.root], ball - root)[:2])
        self.lock = threading.RLock()
        self.busy = False
        self.active = None
        self.joints = self.odom = self.ball = self.state = self.owner = None
        group = ReentrantCallbackGroup()
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.publisher = self.create_publisher(JointCommand, "kick_motor_goals", 10)
        self.create_subscription(JointState, "joint_states", self._joints, 10, callback_group=group)
        self.create_subscription(Odometry, "pelvis_odometry", self._odom, 10, callback_group=group)
        self.create_subscription(
            PoseWithCovarianceStamped, "ball_position_relative_filtered", self._ball, 10, callback_group=group
        )
        self.create_subscription(RobotControlState, "robot_state", self._state, 10, callback_group=group)
        self.create_subscription(Bool, "walking_released", self._owner, 10, callback_group=group)
        self.server = ActionServer(
            self,
            Kick,
            "rl_kick",
            self._execute,
            goal_callback=self._goal,
            cancel_callback=self._cancel,
            callback_group=group,
        )
        self.timer = self.create_timer(self.runtime.dt, self._tick, callback_group=group)
        self.last_failure = None

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def stamp(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _joints(self, msg):
        with self.lock:
            try:
                q, dq = self.runtime.named_joints(msg.name, msg.position, msg.velocity)
                self.joints = (self.stamp(msg), q, dq)
            except ValueError:
                self.joints = None

    def _odom(self, msg):
        with self.lock:
            self.odom = msg

    def _state(self, msg):
        with self.lock:
            self.state = (self.now(), msg.state)

    def _owner(self, msg):
        with self.lock:
            self.owner = (self.now(), msg.data)

    def _ball(self, msg):
        with self.lock:
            try:
                p = msg.pose.pose.position
                position = vector([p.x, p.y, p.z], 3, "ball center")
                if msg.header.frame_id != "odom":
                    transform = self.tf.lookup_transform("odom", msg.header.frame_id, Time.from_msg(msg.header.stamp))
                    t = transform.transform.translation
                    position = forward_rotate(quaternion(transform.transform.rotation), position)
                    position += vector([t.x, t.y, t.z], 3, "TF translation")
                self.ball = (self.stamp(msg), position)
            except Exception as error:
                self.last_failure = str(error)
                self.ball = None

    def _snapshot(self):
        now = self.now()
        if any(x is None for x in (self.joints, self.odom, self.ball, self.state, self.owner)):
            raise ValueError("Required state or walking release is missing")
        joint_stamp, q, dq = self.joints
        for name, stamp in (("joints", joint_stamp), ("odom", self.stamp(self.odom)), ("ball", self.ball[0])):
            try:
                fresh_stamp(now, stamp, self.cfg["sensor_max_age"])
            except ValueError as error:
                raise ValueError(f"{name} timestamp age={now - stamp:.6f}: {error}") from error
        for stamp in (self.state[0], self.owner[0]):
            fresh_stamp(now, stamp, self.cfg["state_max_age"])
        if self.state[1] not in (RobotControlState.CONTROLLABLE, RobotControlState.KICKING):
            raise ValueError("HCM state does not permit kicking")
        if not self.owner[1]:
            raise ValueError("Walking has not released leg ownership")
        odom = self.odom
        if odom.header.frame_id != "odom" or odom.child_frame_id != "pelvis_link":
            raise ValueError("Odometry must describe pelvis_link in odom")
        p = odom.pose.pose.position
        quat = quaternion(odom.pose.pose.orientation)
        angular = odom.twist.twist.angular
        world_angular = forward_rotate(quat, [angular.x, angular.y, angular.z])
        return dict(
            root_pos=vector([p.x, p.y, p.z], 3, "pelvis"),
            root_quat=quat,
            root_ang_vel=world_angular,
            joint_pos=q,
            joint_vel=dq,
            ball_pos=self.ball[1],
        )

    def _request(self, request):
        values = [request.x, request.y, request.strength, request.timeout]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite kick request")
        # The actor has no speed command; this prototype exposes an explicit fixed-policy mode.
        if request.strength != 0:
            raise ValueError("Only fixed-policy mode is supported; strength must be zero")
        if math.hypot(request.x, request.y) < 1e-12:
            raise ValueError("Kick direction is zero")
        if abs(math.atan2(request.y, request.x)) > self.cfg["heading_tolerance"]:
            raise ValueError("Requested heading is outside the validated forward-only prototype")
        state = self._snapshot()
        ball_body = inverse_rotate(state["root_quat"], state["ball_pos"] - state["root_pos"])[:2]
        distances = np.linalg.norm(np.asarray(self.nominal_body) - ball_body, axis=1)
        selected = int(np.argmin(distances))
        if distances[selected] > self.cfg["max_ball_reference_distance"]:
            raise ValueError("Ball is outside the configured reference neighborhood")
        ref = self.references[selected]
        # Passive observed joints cannot be positioned by the policy or preparation controller.
        active = self.runtime.ids
        if (
            np.max(np.abs(state["joint_pos"][active] - ref.data["joint_pos"][0, active]))
            > self.cfg["start_joint_error"]
        ):
            raise ValueError("Startup pose is not reference-compatible")
        if np.max(np.abs(state["joint_vel"])) > self.cfg["start_joint_speed"]:
            raise ValueError("Startup joint motion is too fast")
        expected_q = ref.data["body_quat_w"][0, ref.root]
        for q in (state["root_quat"], expected_q):
            if np.linalg.norm(q) < 1e-12:
                raise ValueError("Invalid root orientation")
        current = forward_rotate(state["root_quat"], [1, 0, 0])
        expected = forward_rotate(expected_q, [1, 0, 0])
        yaw_error = math.atan2(current[1], current[0]) - math.atan2(expected[1], expected[0])
        if abs(math.atan2(math.sin(yaw_error), math.cos(yaw_error))) > self.cfg["start_yaw_error"]:
            raise ValueError("Arbitrary world yaw is not yet validated")
        tilt = inverse_rotate(state["root_quat"], [0, 0, -1])
        expected_tilt = inverse_rotate(expected_q, [0, 0, -1])
        if np.linalg.norm(tilt - expected_tilt) > self.cfg["start_tilt_error"]:
            raise ValueError("Startup tilt is not reference-compatible")
        goal = anchored_goal(
            [request.x, request.y],
            state["root_quat"],
            state["root_pos"],
            self.cfg["goal_distance"],
            self.cfg["goal_height"],
        )
        timeout = request.timeout if request.timeout > 0 else self.cfg["execution_seconds"]
        return ref, goal, timeout

    def _goal(self, request):
        with self.lock:
            if self.busy:
                return GoalResponse.REJECT
            try:
                self._request(request)
            except ValueError as error:
                self.last_failure = str(error)
                self.get_logger().info("Rejected kick: " + str(error))
                return GoalResponse.REJECT
            self.busy = True
            return GoalResponse.ACCEPT

    def _cancel(self, handle):
        return CancelResponse.ACCEPT

    def _finish(self, reason):
        if self.active is not None:
            session = self.active
            self.active = None
            session["reason"] = reason
            session["done"].set_result(reason)

    async def _execute(self, handle):
        session = None
        try:
            with self.lock:
                ref, goal, timeout = self._request(handle.request)
                self.runtime.reset()
                session = dict(
                    handle=handle,
                    ref=ref,
                    goal=goal,
                    timeout=timeout,
                    start=self.now(),
                    start_ns=self.get_clock().now().nanoseconds,
                    previous=None,
                    done=Future(),
                    reason=None,
                )
                self.active = session
                self.timer.reset()
                self._tick()
            reason = await session["done"]
            result = Kick.Result()
            if reason == "complete":
                handle.succeed()
                result.result = Kick.Result.SUCCESS
            elif reason == "cancel":
                handle.canceled()
                result.result = Kick.Result.ABORTED
            else:
                handle.abort()
                result.result = Kick.Result.ABORTED
            return result
        except Exception as error:
            self.last_failure = str(error)
            handle.abort()
            result = Kick.Result()
            result.result = Kick.Result.ABORTED
            return result
        finally:
            with self.lock:
                if self.active is session:
                    self.active = None
                self.busy = False

    def _tick(self):
        with self.lock:
            s = self.active
            if s is None:
                return
            try:
                if s["handle"].is_cancel_requested:
                    self._finish("cancel")
                    return
                now_ns = self.get_clock().now().nanoseconds
                now = now_ns * 1e-9
                if now < s["start"] or (s["previous"] is not None and now < s["previous"]):
                    raise ValueError("Clock moved backwards")
                elapsed = (now_ns - s["start_ns"]) / 1e9
                if s["timeout"] < self.cfg["execution_seconds"] and elapsed >= s["timeout"]:
                    self._finish("timeout")
                    return
                if elapsed >= self.cfg["execution_seconds"]:
                    self._finish("complete")
                    return
                if elapsed >= s["timeout"]:
                    self._finish("timeout")
                    return
                state = self._snapshot()
                obs = self.runtime.observation(**state, **s["ref"].sample(elapsed), goal_pos=s["goal"])
                action = self.runtime.step(obs)
                target = self.runtime.targets(action)
                command = JointCommand()
                command.header.stamp = self.get_clock().now().to_msg()
                command.joint_names = list(target.joint_names)
                command.positions = target.positions.tolist()
                command.velocities = [-1.0] * len(target.joint_names)
                command.accelerations = [-1.0] * len(target.joint_names)
                command.max_torques = target.effort_limits.tolist()
                command.kp = target.kp.tolist()
                command.kd = target.kd.tolist()
                self.publisher.publish(command)
                s["previous"] = now
                feedback = Kick.Feedback()
                feedback.time_remaining = max(0.0, min(s["timeout"], self.cfg["execution_seconds"]) - elapsed)
                s["handle"].publish_feedback(feedback)
            except Exception as error:
                self.last_failure = str(error)
                self._finish("invalid_state")

    def destroy_node(self):
        with self.lock:
            self._finish("shutdown")
        self.server.destroy()
        return super().destroy_node()


def main():
    rclpy.init()
    node = KickNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
