"""Autonomy admission, stale state, angle wrapping, and native DSD integration."""

import threading
import unittest
from types import SimpleNamespace
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.action import GoalResponse, CancelResponse
from bitbots_msgs.action import Kick
from geometry_msgs.msg import PoseStamped
from dynamic_stack_decider import DSD
from bitbots_body_behavior import behavior_dsd
from .handoff import Phase
from .supervised_kick import SupervisedKick
from .autonomy_session import register_r1_elements, HERE, cancel_abandoned_kick, normal_cycle_passed
from .autonomy_dsd.decisions.r1_ball_available import R1BallAvailable
from .autonomy_dsd.decisions.r1_operational import R1Operational
from .autonomy_dsd.decisions.r1_at_approach import R1AtApproach


class TestAdmission(unittest.TestCase):
    def setUp(self):
        self.session = SimpleNamespace(
            router=SimpleNamespace(gate=SimpleNamespace(phase=Phase.READY)),
            adapter=SimpleNamespace(cfg={"heading_tolerance": 0.01}),
            cancel=threading.Event(),
        )
        self.server = SimpleNamespace(session=self.session, reserved=False)

    def request(self, **overrides):
        values = dict(x=1.0, y=0.0, strength=0.0, timeout=-1.0)
        values.update(overrides)
        return Kick.Goal(**values)

    def test_valid_goal_reserves_and_clears_only_previous_cancellation(self):
        self.session.cancel.set()
        self.assertEqual(SupervisedKick.goal(self.server, self.request()), GoalResponse.ACCEPT)
        self.assertTrue(self.server.reserved)
        self.assertFalse(self.session.cancel.is_set())
        self.assertEqual(SupervisedKick.cancel(self.server, None), CancelResponse.ACCEPT)
        self.assertTrue(self.session.cancel.is_set())

    def test_busy_and_faulted_sessions_reject(self):
        self.server.reserved = True
        self.assertEqual(SupervisedKick.goal(self.server, self.request()), GoalResponse.REJECT)
        self.server.reserved = False
        for phase in (Phase.FAULT, Phase.KICKING, Phase.RECOVERING):
            self.session.router.gate.phase = phase
            self.assertEqual(SupervisedKick.goal(self.server, self.request()), GoalResponse.REJECT)

    def test_unsupported_goals_reject_without_reserving(self):
        for fields in (
            dict(x=np.nan),
            dict(strength=2.0),
            dict(timeout=1.0),
            dict(x=0.0, y=0.0),
            dict(x=-1.0),
            dict(y=0.1),
        ):
            self.assertEqual(
                SupervisedKick.goal(self.server, self.request(**fields)),
                GoalResponse.REJECT,
            )
            self.assertFalse(self.server.reserved)


class TestFreshness(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.ctx = SimpleNamespace(
            ball_time=10.0,
            ball_valid=True,
            fault=None,
            enabled=True,
            localization_valid=True,
            status_time=10.0,
            status={"phase": "ready"},
            approach={},
        )
        self.board = SimpleNamespace(
            r1=self.ctx,
            config={
                "r1": dict(
                    ball_max_age=0.25,
                    warmup_seconds=3.0,
                    approach_distance=0.75,
                    approach_position_tolerance=0.025,
                    approach_yaw_tolerance=0.02,
                    approach_dwell=0.2,
                )
            },
            node=SimpleNamespace(
                get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=round(self.now * 1e9)))
            ),
            world_model=SimpleNamespace(ball_has_been_seen=lambda: True),
        )
        self.decision = SimpleNamespace(blackboard=self.board)

    def test_stale_future_and_invalid_ball_cannot_trigger_approach(self):
        self.assertEqual(R1BallAvailable.perform(self.decision), "YES")
        for stamp in (9.7, 10.1):
            self.ctx.ball_time = stamp
            self.assertEqual(R1BallAvailable.perform(self.decision), "NO")
        self.ctx.ball_time = 10.0
        self.ctx.ball_valid = False
        self.assertEqual(R1BallAvailable.perform(self.decision), "NO")

    def test_stale_status_and_missing_localization_hold(self):
        self.assertEqual(R1Operational.perform(self.decision), "RUN")
        for stamp in (9.6, 10.1):
            self.ctx.status_time = stamp
            self.assertEqual(R1Operational.perform(self.decision), "HOLD")
        self.ctx.status_time = 10.0
        self.ctx.localization_valid = False
        self.assertEqual(R1Operational.perform(self.decision), "HOLD")
        self.ctx.fault = "recovery_failed"
        self.assertEqual(R1Operational.perform(self.decision), "FAULT")

    def test_approach_heading_wrap_and_continuous_dwell(self):
        goal = PoseStamped()
        desired = -np.pi + 0.005
        goal.pose.orientation.w = float(np.cos(desired / 2))
        goal.pose.orientation.z = float(np.sin(desired / 2))
        self.board.pathfinding = SimpleNamespace(get_ball_goal=lambda *args: goal)
        self.board.world_model.get_current_position = lambda: (0.0, 0.0, np.pi - 0.005)
        decision = SimpleNamespace(blackboard=self.board, since=None)
        self.assertEqual(R1AtApproach.perform(decision), "NO")
        self.now = 10.21
        self.assertEqual(R1AtApproach.perform(decision), "YES")
        goal.pose.position.x = 0.1
        self.assertEqual(R1AtApproach.perform(decision), "NO")
        goal.pose.position.x = 0.0
        self.now = 10.3
        self.assertEqual(R1AtApproach.perform(decision), "NO")


class TestInterruption(unittest.TestCase):
    def test_real_dsd_interrupt_cancels_abandoned_action(self):
        import tempfile
        from unittest.mock import Mock
        from dynamic_stack_decider.abstract_action_element import AbstractActionElement

        class RootAction(AbstractActionElement):
            def perform(self, reevaluate=False):
                pass

        class FakeKick:
            on_pop = Mock()

        capsule = SimpleNamespace(is_currently_kicking=True, stop_rl_kick=Mock(return_value=True))
        board = SimpleNamespace(
            kick=capsule,
            node=SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: object())),
        )
        dsd = DSD(board)
        dsd.actions = {"RootAction": RootAction, "R1Kick": FakeKick}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dsd") as file:
            file.write("-->\n@RootAction\n")
            file.flush()
            dsd.load_behavior(file.name)
        dsd.stack.append((object(), FakeKick()))
        self.assertFalse(cancel_abandoned_kick(dsd))
        dsd.interrupt()
        FakeKick.on_pop.assert_not_called()
        self.assertTrue(cancel_abandoned_kick(dsd))
        capsule.stop_rl_kick.assert_called_once()
        capsule.is_currently_kicking = False
        self.assertFalse(cancel_abandoned_kick(dsd))


class TestCycleCompletion(unittest.TestCase):
    def setUp(self):
        self.session = SimpleNamespace(
            context=SimpleNamespace(completed=1, search_ticks=75, fault=None),
            router=SimpleNamespace(gate=SimpleNamespace(phase=Phase.READY, owner="walk")),
            seconds=24.0,
            unsafe=lambda: False,
        )
        self.shot = dict(recovered=True, shot_metrics=dict(foot_contact=True), policy_commands_after_terminal=0)

    def test_healthy_complete_resume_window_passes(self):
        self.assertTrue(normal_cycle_passed(self.session, self.shot, 0.3, 20.0))

    def test_fault_after_recovery_and_sufficient_distance_cannot_pass(self):
        # Matches the observed large-ball failure: kick recovered, distance
        # exceeded 0.15 m, then ownership faulted before the four-second window.
        self.session.seconds = 21.441
        self.session.router.gate.phase = Phase.FAULT
        self.session.router.gate.owner = "hcm"
        self.session.unsafe = lambda: True
        self.assertFalse(normal_cycle_passed(self.session, self.shot, 0.7397, 20.181))

    def test_incomplete_window_cannot_pass_even_with_enough_distance(self):
        self.assertFalse(normal_cycle_passed(self.session, self.shot, 0.3, 20.02))
        self.assertFalse(normal_cycle_passed(self.session, self.shot, 0.3, None))

    def test_unsafe_final_sample_before_next_fault_tick_cannot_pass(self):
        self.session.unsafe = lambda: True
        self.assertFalse(normal_cycle_passed(self.session, self.shot, 0.3, 20.0))

    def test_late_commands_or_behavior_fault_cannot_pass(self):
        self.shot["policy_commands_after_terminal"] = 1
        self.assertFalse(normal_cycle_passed(self.session, self.shot, 0.3, 20.0))
        self.shot["policy_commands_after_terminal"] = 0
        self.session.context.fault = "late_motion_fault"
        self.assertFalse(normal_cycle_passed(self.session, self.shot, 0.3, 20.0))


class TestNativeDSD(unittest.TestCase):
    def test_custom_tree_preserves_native_actions(self):
        rclpy.init()
        node = Node(
            "r1_dsd_parse_test",
            parameter_overrides=[Parameter("r1.approach_distance", value=0.75)],
            automatically_declare_parameters_from_overrides=True,
        )
        try:
            dsd = DSD(object(), node=node)
            dsd.register_actions(behavior_dsd.actions.__path__[0])
            dsd.register_decisions(behavior_dsd.decisions.__path__[0])
            register_r1_elements(dsd)
            self.assertIn("Stand", dsd.actions)
            self.assertIn("GoToBall", dsd.actions)
            self.assertIn("R1Kick", dsd.actions)
            dsd.load_behavior(str(HERE / "autonomy_dsd/r1_autonomous.dsd"))
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
