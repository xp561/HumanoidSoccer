"""Simulation-only cmd_vel walker; the shared router owns motor output."""

import os
import numpy as np
from rclpy.node import Node
from bitbots_msgs.msg import JointCommand
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from .walking_runtime import WalkingRuntime


class WalkingNode(Node):
    def __init__(self, *, isolated_domain=179, **kwargs):
        super().__init__("r1_basic_walker", namespace="/r1_adapter_test", **kwargs)
        if self.get_namespace() != "/r1_adapter_test" or os.environ.get(
            "ROS_DOMAIN_ID"
        ) != str(isolated_domain):
            raise RuntimeError(
                "Walker requires isolated simulation domain and namespace"
            )
        self.runtime = WalkingRuntime()
        self.joints = self.odom = None
        self.command = np.zeros(3)
        self.command_ns = None
        self.command_error = None
        self.output = self.create_publisher(JointCommand, "walking_motor_goals", 10)
        self.create_subscription(JointState, "joint_states", self.on_joints, 10)
        self.create_subscription(Odometry, "pelvis_odometry", self.on_odom, 10)
        self.create_subscription(Twist, "cmd_vel", self.on_command, 10)

    @staticmethod
    def stamp_ns(msg):
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def on_joints(self, msg):
        self.joints = msg

    def on_odom(self, msg):
        self.odom = msg

    def on_command(self, msg):
        if (
            msg.angular.x == -1
            and msg.angular.y
            == msg.angular.z
            == msg.linear.x
            == msg.linear.y
            == msg.linear.z
            == 0
        ):
            self.command[:] = 0
            self.command_ns = self.get_clock().now().nanoseconds
            self.command_error = None
            return
        values = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        bounds = self.runtime.contract["training_command_limits"]
        valid = np.isfinite(values).all() and all(
            bounds[n][0] - 1e-6 <= values[i] <= bounds[n][1] + 1e-6
            for i, n in enumerate(("vx", "vy", "wz"))
        )
        valid = valid and msg.linear.z == msg.angular.x == msg.angular.y == 0
        self.command = values if valid else np.zeros(3)
        self.command_ns = self.get_clock().now().nanoseconds
        self.command_error = None if valid else "invalid_cmd_vel; zero command selected"

    def ready(self, ns):
        return (
            self.joints is not None
            and self.odom is not None
            and self.stamp_ns(self.joints) == self.stamp_ns(self.odom) == ns
            and self.get_clock().now().nanoseconds == ns
        )

    def publish_target(self, ns, *, stopping=False):
        if not self.ready(ns):
            raise ValueError("Walker sensors do not match current simulation step")
        if (
            self.odom.header.frame_id != "odom"
            or self.odom.child_frame_id != "pelvis_link"
        ):
            raise ValueError("Walking odometry must describe pelvis_link in odom")
        command = self.command.copy()
        if (
            stopping
            or self.command_ns is None
            or not 0 <= ns - self.command_ns <= 250_000_000
        ):
            command[:] = 0
        pose = self.odom.pose.pose.orientation
        angular = self.odom.twist.twist.angular
        q = np.array([pose.w, pose.x, pose.y, pose.z])
        body = np.array([angular.x, angular.y, angular.z])
        # Odometry angular velocity is already child/body-frame; runtime expects world-frame.
        world = self.runtime.inverse_rotate(q * np.array([1, -1, -1, -1]), body)
        rt = self.runtime
        target = rt.target(
            joint_names=self.joints.name,
            joint_pos=self.joints.position,
            joint_vel=self.joints.velocity,
            root_quat=q,
            root_ang_vel=world,
            command=command,
        )
        msg = JointCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = list(rt.names)
        msg.positions = target.tolist()
        msg.kp = rt.kp.tolist()
        msg.kd = rt.kd.tolist()
        msg.max_torques = rt.effort_limit.tolist()
        self.output.publish(msg)
        self.last_applied_command = command.copy()
        return command
