"""Bit-Bots velocity command semantics and watchdog on the actual ROS callback."""

import unittest
from types import SimpleNamespace
import numpy as np
from geometry_msgs.msg import Twist
from .walking_ros import WalkingNode
from .walking_runtime import WalkingRuntime


class TestWalkingCommands(unittest.TestCase):
    def setUp(self):
        self.node = SimpleNamespace(
            runtime=WalkingRuntime(),
            command=np.ones(3),
            command_ns=None,
            command_error=None,
            get_clock=lambda: SimpleNamespace(
                now=lambda: SimpleNamespace(nanoseconds=123)
            ),
        )

    def test_bitbots_stop_sentinel(self):
        msg = Twist()
        msg.angular.x = -1.0
        WalkingNode.on_command(self.node, msg)
        np.testing.assert_array_equal(self.node.command, [0, 0, 0])
        self.assertIsNone(self.node.command_error)
        self.assertEqual(self.node.command_ns, 123)

    def test_invalid_command_stops_instead_of_reusing_old_target(self):
        for vx in (float("nan"), 0.41):
            msg = Twist()
            msg.linear.x = vx
            WalkingNode.on_command(self.node, msg)
            np.testing.assert_array_equal(self.node.command, [0, 0, 0])
            self.assertIsNotNone(self.node.command_error)

    def test_body_velocity_command(self):
        msg = Twist()
        msg.linear.x = 0.2
        msg.linear.y = -0.1
        msg.angular.z = 0.3
        WalkingNode.on_command(self.node, msg)
        np.testing.assert_array_equal(self.node.command, [0.2, -0.1, 0.3])
        self.assertIsNone(self.node.command_error)


if __name__ == "__main__":
    unittest.main()
