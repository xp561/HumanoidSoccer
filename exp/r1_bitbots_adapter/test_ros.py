"""Native ROS Action tests against a synthetic state source and command sink."""

import json
import os
import threading
import time
import unittest
from pathlib import Path

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from bitbots_msgs.action import Kick
from bitbots_msgs.msg import JointCommand, RobotControlState

from .ros_node import KickNode, forward_rotate


def wait(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Timed out waiting for ROS event")


def future_result(future):
    wait(future.done)
    return future.result()


class Fixture(Node):
    def __init__(self, adapter):
        super().__init__("r1_synthetic_sensors", namespace="/r1_adapter_test")
        self.adapter = adapter
        self.lock = threading.RLock()
        self.commands = []
        self.feedback = []
        self.publish_enabled = True
        self.release = True
        self.hcm_state = RobotControlState.CONTROLLABLE
        self.ball_age = 0.0
        self.missing_head = False
        self.ball_frame = "odom"
        self.bad_pose = False
        self.publishers_by_name = {
            "joint_states": self.create_publisher(JointState, "joint_states", 10),
            "pelvis_odometry": self.create_publisher(Odometry, "pelvis_odometry", 10),
            "ball_position_relative_filtered": self.create_publisher(
                PoseWithCovarianceStamped, "ball_position_relative_filtered", 10
            ),
            "robot_state": self.create_publisher(RobotControlState, "robot_state", 10),
            "walking_released": self.create_publisher(Bool, "walking_released", 10),
        }
        self.create_subscription(
            JointCommand, "kick_motor_goals", lambda msg: self.commands.append((time.monotonic(), msg)), 100
        )
        self.client = ActionClient(self, Kick, "rl_kick")
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "odom"
        transform.child_frame_id = "test_ball_frame"
        transform.transform.translation.x = 0.3
        transform.transform.translation.y = -0.2
        transform.transform.rotation.z = np.sin(np.pi / 4)
        transform.transform.rotation.w = np.cos(np.pi / 4)
        self.tf_broadcaster.sendTransform(transform)
        self.timer = self.create_timer(0.01, self.publish)

    def publish(self):
        with self.lock:
            if not self.publish_enabled:
                return
            stamp = self.get_clock().now().to_msg()
            ref = self.adapter.references[0]
            joint = JointState()
            joint.header.stamp = stamp
            joint.name = list(self.adapter.runtime.names[::-1])
            joint.position = ref.data["joint_pos"][0, ::-1].astype(float).tolist()
            joint.velocity = [0.0] * len(joint.name)
            if self.bad_pose:
                joint.position[0] += 1.0
            if self.missing_head:
                index = joint.name.index("head_pitch_joint")
                names = list(joint.name)
                positions = list(joint.position)
                velocities = list(joint.velocity)
                del names[index], positions[index], velocities[index]
                joint.name, joint.position, joint.velocity = names, positions, velocities
            self.publishers_by_name["joint_states"].publish(joint)
            root = ref.data["body_pos_w"][0, ref.root]
            quat = ref.data["body_quat_w"][0, ref.root]
            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = "odom"
            odom.child_frame_id = "pelvis_link"
            odom.pose.pose.position.x = float(root[0])
            odom.pose.pose.position.y = float(root[1])
            odom.pose.pose.position.z = float(root[2])
            odom.pose.pose.orientation.w = float(quat[0])
            odom.pose.pose.orientation.x = float(quat[1])
            odom.pose.pose.orientation.y = float(quat[2])
            odom.pose.pose.orientation.z = float(quat[3])
            self.publishers_by_name["pelvis_odometry"].publish(odom)
            ball = PoseWithCovarianceStamped()
            ball.header.stamp = rclpy.time.Time(
                nanoseconds=self.get_clock().now().nanoseconds - int(self.ball_age * 1e9)
            ).to_msg()
            ball.header.frame_id = self.ball_frame
            xy = self.adapter.runtime.meta["motion_nominal_ball_xy"][0]
            position = np.array([*xy, self.adapter.cfg["goal_height"]])
            if self.ball_frame == "test_ball_frame":
                position = forward_rotate(
                    [np.cos(np.pi / 4), 0, 0, -np.sin(np.pi / 4)], position - np.array([0.3, -0.2, 0])
                )
            ball.pose.pose.position.x = float(position[0])
            ball.pose.pose.position.y = float(position[1])
            ball.pose.pose.position.z = float(position[2])
            self.publishers_by_name["ball_position_relative_filtered"].publish(ball)
            self.publishers_by_name["robot_state"].publish(RobotControlState(state=self.hcm_state))
            self.publishers_by_name["walking_released"].publish(Bool(data=self.release))


class TestROS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("ROS_DOMAIN_ID"):
            raise RuntimeError("An isolated domain must be set by the test launcher")
        rclpy.init()
        cls.adapter = KickNode(
            parameter_overrides=[
                Parameter("execution_seconds", value=0.3),
            ]
        )
        cls.fixture = Fixture(cls.adapter)
        cls.executor = SingleThreadedExecutor()
        cls.executor.add_node(cls.adapter)
        cls.executor.add_node(cls.fixture)
        cls.thread = threading.Thread(target=cls.executor.spin, daemon=True)
        cls.thread.start()
        if not cls.fixture.client.wait_for_server(timeout_sec=10):
            raise AssertionError("Action server unavailable")

    @classmethod
    def tearDownClass(cls):
        wait(lambda: not cls.adapter.busy)
        cls.executor.shutdown(timeout_sec=3)
        cls.thread.join(timeout=3)
        cls.fixture.destroy_node()
        cls.adapter.destroy_node()
        rclpy.shutdown()

    def setUp(self):
        wait(lambda: not self.adapter.busy)
        f = self.fixture
        with f.lock:
            f.publish_enabled = True
            f.release = True
            f.hcm_state = RobotControlState.CONTROLLABLE
            f.ball_age = 0.0
            f.missing_head = False
            f.bad_pose = False
            f.ball_frame = "odom"
        time.sleep(0.12)
        with self.adapter.lock:
            self.adapter._snapshot()
        f.commands.clear()
        f.feedback.clear()

    def send(self, **overrides):
        request = Kick.Goal()
        request.x, request.y, request.strength, request.timeout = 1.0, 0.0, 0.0, -1.0
        for name, value in overrides.items():
            setattr(request, name, float(value))
        return future_result(
            self.fixture.client.send_goal_async(
                request, feedback_callback=lambda msg: self.fixture.feedback.append(msg.feedback.time_remaining)
            )
        )

    def assert_quiet(self):
        time.sleep(0.06)
        count = len(self.fixture.commands)
        time.sleep(0.08)
        self.assertEqual(len(self.fixture.commands), count)

    def test_passive_head_is_observed_but_not_a_preparation_target(self):
        with self.adapter.lock:
            stamp, q, dq = self.adapter.joints
            original = q.copy()
            for name in ("head_pitch_joint", "head_yaw_joint"):
                q[self.adapter.runtime.names.index(name)] += 0.3
            try:
                request = Kick.Goal(x=1.0, y=0.0, strength=0.0, timeout=-1.0)
                self.adapter._request(request)
            finally:
                q[:] = original

    def test_bitbots_fixed_mode_capsule_completes_real_action(self):
        from types import SimpleNamespace
        from bitbots_blackboard.capsules.kick_capsule import KickCapsule

        quat = self.adapter.references[0].data["body_quat_w"][0, self.adapter.references[0].root]
        forward = forward_rotate(quat, [1, 0, 0])
        yaw = float(np.arctan2(forward[1], forward[0]))
        board = SimpleNamespace(
            config={"rl_kick": {"timeout": -1.0, "strength_mode": "fixed"}},
            world_model=SimpleNamespace(get_current_position=lambda: (0.0, 0.0, yaw)),
        )
        capsule = KickCapsule(self.fixture, board)
        try:
            self.assertTrue(capsule.start_rl_kick(float(np.degrees(yaw))))
            wait(lambda: not capsule.is_currently_kicking)
            self.assertEqual(capsule.last_rl_kick_result, Kick.Result.SUCCESS)
            self.assertEqual(capsule.last_rl_kick_status, GoalStatus.STATUS_SUCCEEDED)
            self.assertGreater(len(self.fixture.commands), 0)
            self.assert_quiet()
            self.assertFalse(capsule.start_rl_kick(float(np.degrees(yaw)), 2.0))
            self.assertEqual(capsule.last_rl_kick_result, Kick.Result.REJECTED)
            self.assert_quiet()
        finally:
            capsule._rl_kick_client.destroy()

    def test_complete_and_joint_command_serialization(self):
        self.assert_quiet()
        goal = self.send()
        self.assertTrue(goal.accepted, self.adapter.last_failure)
        result = future_result(goal.get_result_async())
        self.assertEqual(result.status, GoalStatus.STATUS_SUCCEEDED)
        self.assertEqual(result.result.result, Kick.Result.SUCCESS)
        self.assertGreaterEqual(len(self.fixture.commands), 10)
        stamps = np.array([msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 for _, msg in self.fixture.commands])
        periods = np.diff(stamps)
        self.assertLess(abs(float(np.median(periods)) - self.adapter.runtime.dt), 0.005)
        self.assertLess(float(periods.max()), 0.08)
        self.assertGreater(len(self.fixture.feedback), 0)
        self.assertTrue(np.isfinite(self.fixture.feedback).all())
        self.assertTrue(np.all(np.diff(self.fixture.feedback) <= 0))
        output = Path("logs/r1_logic_fix/r1_ros_action_20261023")
        output.mkdir(parents=True, exist_ok=True)
        (output / "command_timing.json").write_text(
            json.dumps(
                dict(
                    commands=len(stamps),
                    expected_period=self.adapter.runtime.dt,
                    median_period=float(np.median(periods)),
                    maximum_period=float(periods.max()),
                    minimum_period=float(periods.min()),
                    feedback_count=len(self.fixture.feedback),
                ),
                indent=2,
            )
        )
        for _, command in self.fixture.commands:
            self.assertEqual(command.joint_names, list(self.adapter.runtime.action_names))
            self.assertEqual(len(command.positions), 24)
            self.assertTrue(np.isfinite(command.positions).all())
            np.testing.assert_array_equal(command.kp, self.adapter.runtime.kp)
            np.testing.assert_array_equal(command.kd, self.adapter.runtime.kd)
            self.assertGreater(command.header.stamp.sec, 0)
        self.assert_quiet()

    def test_cancel_stops_output(self):
        goal = self.send()
        self.assertTrue(goal.accepted, self.adapter.last_failure)
        wait(lambda: len(self.fixture.commands) >= 2)
        cancel = future_result(goal.cancel_goal_async())
        self.assertTrue(cancel.goals_canceling)
        result = future_result(goal.get_result_async())
        self.assertEqual(result.status, GoalStatus.STATUS_CANCELED)
        self.assert_quiet()

    def test_busy_request_rejected(self):
        first = self.send()
        self.assertTrue(first.accepted, self.adapter.last_failure)
        second = self.send()
        self.assertFalse(second.accepted)
        future_result(first.cancel_goal_async())
        future_result(first.get_result_async())

    def test_unsupported_requests_rejected(self):
        for kwargs in (dict(strength=3), dict(x=0, y=0), dict(x=0, y=1), dict(x=float("nan"))):
            with self.subTest(request=kwargs):
                self.assertFalse(self.send(**kwargs).accepted)
        self.assertFalse(self.fixture.commands)

    def test_missing_head_rejected(self):
        self.fixture.missing_head = True
        time.sleep(0.06)
        self.assertFalse(self.send().accepted)
        self.assertFalse(self.fixture.commands)

    def test_bad_start_pose_rejected(self):
        self.fixture.bad_pose = True
        time.sleep(0.06)
        self.assertFalse(self.send().accepted)

    def test_owner_required_before_start(self):
        self.fixture.release = False
        time.sleep(0.06)
        self.assertFalse(self.send().accepted)
        self.assertFalse(self.fixture.commands)

    def test_owner_loss_aborts(self):
        goal = self.send()
        self.assertTrue(goal.accepted, self.adapter.last_failure)
        wait(lambda: len(self.fixture.commands) >= 2)
        self.fixture.release = False
        result = future_result(goal.get_result_async())
        self.assertEqual(result.status, GoalStatus.STATUS_ABORTED)
        self.assert_quiet()

    def test_hcm_fall_aborts(self):
        goal = self.send()
        self.assertTrue(goal.accepted, self.adapter.last_failure)
        wait(lambda: len(self.fixture.commands) >= 2)
        self.fixture.hcm_state = RobotControlState.FALLING
        result = future_result(goal.get_result_async())
        self.assertEqual(result.status, GoalStatus.STATUS_ABORTED)
        self.assert_quiet()

    def test_stale_ball_rejected(self):
        self.fixture.ball_age = 1.0
        time.sleep(0.06)
        self.assertFalse(self.send().accepted)

    def test_timeout_is_not_success(self):
        goal = self.send(timeout=0.06)
        self.assertTrue(goal.accepted, self.adapter.last_failure)
        result = future_result(goal.get_result_async())
        self.assertEqual(result.status, GoalStatus.STATUS_ABORTED)
        self.assertEqual(result.result.result, Kick.Result.ABORTED)
        self.assert_quiet()

    def test_lost_sensor_stream_aborts(self):
        goal = self.send()
        self.assertTrue(goal.accepted, self.adapter.last_failure)
        wait(lambda: len(self.fixture.commands) >= 2)
        self.fixture.publish_enabled = False
        result = future_result(goal.get_result_async())
        self.assertEqual(result.status, GoalStatus.STATUS_ABORTED)
        self.assert_quiet()

    def test_tf_ball_conversion(self):
        self.fixture.ball_frame = "test_ball_frame"
        time.sleep(0.1)
        with self.adapter.lock:
            actual = self.adapter._snapshot()["ball_pos"]
        expected = [*self.adapter.runtime.meta["motion_nominal_ball_xy"][0], self.adapter.cfg["goal_height"]]
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        goal = self.send()
        self.assertTrue(goal.accepted, self.adapter.last_failure)
        self.assertEqual(future_result(goal.get_result_async()).status, GoalStatus.STATUS_SUCCEEDED)


if __name__ == "__main__":
    unittest.main()
