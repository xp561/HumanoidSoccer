"""Restart handshake, stale-state inhibition and explicit supervisor admission."""

import unittest
from types import SimpleNamespace

from game_controller_hsl_interfaces.msg import GameState
from rclpy.action import GoalResponse

from .match_control import MatchControl
from .supervised_kick import SupervisedKick


def message(t, **fields):
    m = GameState()
    m.header.stamp.sec = int(t)
    m.header.stamp.nanosec = round((t - int(t)) * 1e9)
    m.main_state = m.STATE_PLAYING
    m.kicking_team = 42
    for key, value in fields.items():
        setattr(m, key, value)
    return m


def ns(seconds):
    return round(seconds * 1_000_000_000)


class TestMatchControl(unittest.TestCase):
    def test_only_supported_playing_allows_motion(self):
        p = MatchControl()
        m = message(1)
        self.assertFalse(p.permission(m, ns(1)).allowed)
        p.ingest(m, ns(1))
        self.assertTrue(p.permission(m, ns(1)).allowed)
        for state in [m.STATE_INITIAL, m.STATE_READY, m.STATE_SET, m.STATE_FINISHED]:
            m = message(2 + state, main_state=state)
            p.ingest(m, 2 + state)
            self.assertFalse(p.permission(m, 2 + state).allowed)

    def test_stop_penalty_red_card_and_other_restart_hold(self):
        for fields in [
            dict(stopped=True),
            dict(penalized=True),
            dict(penalty=1),
            dict(penalized_in_place=True),
            dict(has_red_card=True),
            dict(game_phase=3),
            dict(set_play=1, kicking_team=43),
        ]:
            with self.subTest(fields=fields):
                p = MatchControl()
                m = message(1, **fields)
                p.ingest(m, ns(1))
                self.assertFalse(p.permission(m, ns(1)).allowed)
        p = MatchControl()
        m = message(1, set_play=1, kicking_team=42)
        p.ingest(m, ns(1))
        self.assertTrue(p.permission(m, ns(1)).allowed)

    def test_stale_future_and_out_of_order_frames_do_not_enable_motion(self):
        p = MatchControl()
        m = message(2)
        p.ingest(m, ns(2))
        self.assertEqual(p.permission(m, ns(3.01)).reason, "stale_gamestate")
        old = message(1.9)
        p.ingest(old, ns(2))
        self.assertFalse(p.permission(old, ns(2)).allowed)
        future = message(5)
        p.ingest(future, ns(2))
        self.assertFalse(p.permission(future, ns(2)).allowed)
        fresh = message(2.1)
        p.ingest(fresh, ns(2.1))
        self.assertTrue(p.permission(fresh, ns(2.1)).allowed)

    def test_conflicting_duplicate_cannot_clear_stop(self):
        p = MatchControl()
        stop = message(1, stopped=True)
        p.ingest(stop, ns(1))
        run = message(1)
        p.ingest(run, ns(1))
        self.assertEqual(
            p.permission(run, ns(1)).reason, "conflicting_gamestate_timestamp"
        )

    def test_terminal_requires_explicit_fresh_set_and_restart_ack(self):
        p = MatchControl()
        m = message(1)
        p.ingest(m, ns(1))
        p.latch_terminal(dict(kind="goal"), ns(1))
        m = message(2)
        p.ingest(m, ns(2))
        self.assertFalse(p.permission(m, ns(2)).allowed)
        self.assertFalse(p.restart_state_valid(m, ns(2)))
        with self.assertRaises(RuntimeError):
            p.acknowledge_restart()
        m = message(3, main_state=m.STATE_SET, stopped=True)
        p.ingest(m, ns(3))
        self.assertFalse(p.restart_state_valid(m, ns(4.01)))
        self.assertTrue(p.restart_state_valid(m, ns(3)))
        p.acknowledge_restart()
        self.assertFalse(p.permission(m, ns(3)).allowed)
        m = message(3.02)
        p.ingest(m, ns(3.02))
        self.assertTrue(p.permission(m, ns(3.02)).allowed)
        self.assertEqual(p.epoch, 1)

    def test_exact_nanoseconds_never_false_stale_from_float_rounding(self):
        p = MatchControl()
        for now in [
            1_261_000_000,
            3_381_000_000,
            8_681_000_000,
            25_941_000_000,
            1_700_000_000_123_456_789,
        ]:
            m = GameState()
            m.header.stamp.sec = now // 1_000_000_000
            m.header.stamp.nanosec = now % 1_000_000_000
            m.main_state = m.STATE_PLAYING
            m.kicking_team = 42
            p.ingest(m, now)
            self.assertTrue(p.permission(m, now).allowed)

    def test_supervisor_rejects_blocked_match_before_reserving(self):
        server = SimpleNamespace(
            session=SimpleNamespace(motion_permitted=lambda: False), reserved=False
        )
        self.assertEqual(SupervisedKick.goal(server, None), GoalResponse.REJECT)
        self.assertFalse(server.reserved)


if __name__ == "__main__":
    unittest.main()
