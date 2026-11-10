"""Scripted simulation referee exercising native GameState and match restarts."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import rclpy
import yaml
from game_controller_hsl_interfaces.msg import GameState
from std_srvs.srv import Trigger
from rclpy.action import ActionClient
from bitbots_msgs.action import Kick

from .autonomy_session import HERE
from .handoff import Phase
from .match_flow_session import GameFlowSession
from .mujoco_bridge import spin_until


class SimReferee:
    cancel_after_commands = 30

    def __init__(self, session, scenario):
        self.scenario = scenario
        self.pub = session.bridge.create_publisher(GameState, "gamestate", 10)
        self.triggered = None
        self.pause_end = None
        self.future = None
        self.restart_done = False
        self.set_started = False
        self.messages = []
        self.bad_request = None
        self.finished_after = None
        self.fault_probe = (
            ActionClient(session.bridge, Kick, "supervised_kick")
            if scenario == "cancel_kick"
            else None
        )
        self.fault_probe_future = None
        self.fault_playing_sent = False
        self.manual_future = None

    def tick(self, session):
        t = session.seconds
        if self.scenario == "no_gamestate":
            return
        msg = GameState()
        msg.header.stamp = session.bridge.stamp(session.ns)
        msg.players_per_team = 4
        msg.competition_type = msg.COMPETITION_TYPE_SMALL
        msg.main_state = (
            msg.STATE_READY if t < 2 else msg.STATE_SET if t < 3 else msg.STATE_PLAYING
        )
        msg.stopped = t < 3
        msg.kicking_team = 42
        msg.seconds_remaining = 600
        kind = self.scenario
        if self.triggered is None and (
            (
                kind
                in (
                    "pause_walk",
                    "stale",
                    "penalty",
                    "opponent_restart",
                    "finished",
                    "manual_stop",
                )
                and t >= 6.5
            )
            or (kind == "cancel_preparation" and session.stage == "preparing")
            or (
                kind == "cancel_kick"
                and session.stage == "kicking"
                and len(session.kick_commands) >= self.cancel_after_commands
            )
        ):
            self.triggered = t
            self.pause_end = t + 2.5
        if self.triggered is not None:
            paused = t < self.pause_end
            if kind in ("cancel_preparation", "cancel_kick", "finished", "manual_stop"):
                paused = True
            if kind == "stale" and paused:
                return
            if paused:
                if kind == "penalty":
                    msg.penalized = True
                    msg.penalty = msg.PENALTY_ILLEGAL_POSITIONING
                elif kind == "opponent_restart":
                    msg.set_play = msg.SET_PLAY_DIRECT_FREE_KICK
                    msg.kicking_team = 43
                elif kind == "finished":
                    msg.main_state = msg.STATE_FINISHED
                else:
                    msg.stopped = True
        if kind == "manual_stop" and self.triggered is not None:
            msg.main_state, msg.stopped = msg.STATE_PLAYING, False
            if self.manual_future is None:
                self.manual_future = session.stop_client.call_async(Trigger.Request())
        if kind == "bad_restart" and t >= 5 and self.bad_request is None:
            self.bad_request = session.restart_client.call_async(Trigger.Request())
        if (
            kind in ("goal_restart", "out_restart")
            and session.match_control.terminal is not None
            and session.match_control.epoch == 0
        ):
            if (
                session.router.gate.phase == Phase.READY
                and not session.supervisor.reserved
                and session.stable_monitor is not None
                and session.stable_monitor.elapsed >= session.config.dwell
            ):
                self.set_started = True
            if self.set_started:
                msg.main_state, msg.stopped = msg.STATE_SET, True
                # A fresh SET must traverse the actual native subscriber before
                # the service can acknowledge a referee placement.
                if (
                    session.match_control.restart_state_valid(
                        session.capsule.gamestate, session.ns
                    )
                    and self.future is None
                ):
                    self.future = session.restart_client.call_async(Trigger.Request())
        if self.future is not None and self.future.done():
            self.restart_done = self.future.result().success
        if (
            kind == "cancel_kick"
            and session.router.gate.phase == Phase.FAULT
            and self.triggered is not None
            and t >= self.triggered + 2
        ):
            msg.main_state = msg.STATE_PLAYING
            msg.stopped = False
            self.fault_playing_sent = True
        self.pub.publish(msg)
        spin_until(
            session.executor,
            lambda: session.capsule.gamestate.header.stamp == msg.header.stamp,
        )
        if self.fault_playing_sent and self.fault_probe_future is None:
            self.fault_probe_future = self.fault_probe.send_goal_async(
                Kick.Goal(x=1.0, y=0.0, strength=0.0, timeout=-1.0)
            )
        self.messages.append(
            dict(
                seconds=t,
                main_state=msg.main_state,
                stopped=msg.stopped,
                penalized=msg.penalized,
                set_play=msg.set_play,
            )
        )

    def done(self, session):
        t = session.seconds
        if self.scenario == "no_gamestate":
            return t >= 5
        if self.scenario in (
            "cancel_preparation",
            "cancel_kick",
            "finished",
            "manual_stop",
        ):
            return (
                self.triggered is not None
                and t >= self.triggered + 5
                and not session.supervisor.reserved
            )
        if self.scenario == "bad_restart":
            return self.bad_request is not None and self.bad_request.done() and t >= 7
        if self.scenario in ("goal_restart", "out_restart"):
            return (
                session.match_control.epoch == 1
                and session.match_control.terminal is not None
                and session.router.gate.phase == Phase.READY
                and not session.supervisor.reserved
                and session.stable_monitor.elapsed >= 0.8
            )
        return (
            session.match_control.terminal is not None
            and session.router.gate.phase == Phase.READY
            and not session.supervisor.reserved
            and session.stable_monitor.elapsed >= 0.8
        )


def verify(session, referee, arena):
    scenario = referee.scenario
    trace = session.flow_trace
    assert trace
    # Even delayed native nonzero navigation commands cannot bypass this fence.
    blocked = [
        r
        for r in trace
        if r["command_owner"] == "walk" and r["walk_permitted_at_command"] is False
    ]
    assert blocked
    assert all(
        r["walker_applied_command"] is not None
        and np.linalg.norm(r["walker_applied_command"]) == 0
        for r in blocked
    )
    assert arena["robot_state_resets_after_initialization"] == 0
    assert all(
        p["robot_qpos_change"] == p["robot_qvel_change"] == 0
        for p in arena["ball_placements"]
    )
    reports = [
        json.loads(p.read_text()) for p in sorted(session.output.glob("trial_*.json"))
    ]
    assert all(r["policy_commands_after_terminal"] == 0 for r in reports)
    if scenario == "no_gamestate":
        assert not session.capsule.received_gamestate()
        assert not reports and len(session.kick_commands) == 0
        assert not session.unsafe()
    elif scenario in ("goal_restart", "out_restart"):
        assert referee.restart_done and len(arena["ball_placements"]) == 1
        assert session.match_control.epoch == 1
        assert len(arena["events"]) == 2
        assert arena["events"][0]["kind"] == (
            "goal" if scenario == "goal_restart" else "out"
        )
        assert arena["events"][1]["kind"] in ("goal", "out")
        if scenario == "goal_restart":
            assert arena["events"][1]["kind"] == "goal"
        assert session.router.gate.phase == Phase.READY and not session.unsafe()
        assert all(
            r["recovered"]
            and r["outcome"] == "policy_completed"
            and r["policy_commands"] == 260
            for r in reports
        )
        assert len(reports) == (2 if scenario == "goal_restart" else 1)
    elif scenario == "bad_restart":
        assert referee.bad_request.done() and not referee.bad_request.result().success
        assert not arena["ball_placements"]
    elif scenario in ("cancel_preparation", "cancel_kick"):
        assert referee.triggered is not None and reports
        if scenario == "cancel_preparation":
            assert sum(r["policy_commands"] for r in reports) == 0
        else:
            count = sum(r["policy_commands"] for r in reports)
            assert (
                referee.cancel_after_commands
                <= count
                <= referee.cancel_after_commands + 2
            )
        assert not arena["ball_placements"]
        assert session.router.gate.owner in ("walk", "hcm")
        if session.router.gate.phase == Phase.READY:
            assert not session.unsafe()
            assert all(r["recovered"] for r in reports)
        if session.router.gate.phase == Phase.FAULT:
            assert session.router.gate.owner == "hcm"
            if scenario == "cancel_kick":
                assert referee.fault_playing_sent and not session.capsule.is_stopped()
                assert (
                    referee.fault_probe_future is not None
                    and referee.fault_probe_future.done()
                )
                assert not referee.fault_probe_future.result().accepted
    elif scenario == "manual_stop":
        assert (
            referee.manual_future is not None
            and referee.manual_future.done()
            and referee.manual_future.result().success
        )
        assert getattr(session, "manual_stop_requested", False)
        assert (
            session.capsule.get_main_state() == GameState.STATE_PLAYING
            and not session.capsule.is_stopped()
        )
        assert not reports and len(session.kick_commands) == 0
        assert not session.unsafe()
        assert all(
            not r["motion_allowed"]
            for r in trace
            if r["seconds"] > referee.triggered + 0.1
        )
    elif scenario == "finished":
        assert not reports and len(session.kick_commands) == 0
        assert any(
            r["reason"] == "main_state_not_playing"
            for r in trace
            if r["seconds"] > referee.triggered
        )
    else:
        expected = dict(
            pause_walk="game_controller_stopped",
            stale="stale_gamestate",
            penalty="player_penalized",
            opponent_restart="opponent_set_play_hold",
        )[scenario]
        window = [
            r
            for r in trace
            if referee.triggered + 1.3 <= r["seconds"] <= referee.pause_end - 0.02
        ]
        assert window and all(not r["motion_allowed"] for r in window)
        assert any(r["reason"] == expected for r in window)
        assert any(
            r["motion_allowed"] for r in trace if r["seconds"] > referee.pause_end
        )
        assert arena["events"] and arena["events"][0]["kind"] in ("goal", "out")
        assert reports and all(
            r["outcome"] == "policy_completed" and r["recovered"] for r in reports
        )
        assert session.router.gate.phase == Phase.READY and not session.unsafe()
    return dict(
        passed=True,
        scenario=scenario,
        seconds=session.seconds,
        native_game_state=session.capsule.get_main_state(),
        native_stopped=session.capsule.is_stopped(),
        completed_actions=session.context.completed,
        policy_commands=len(session.kick_commands),
        shot_reports=[
            dict(
                outcome=r["outcome"],
                recovered=r["recovered"],
                commands=r["policy_commands"],
                late_commands=r["policy_commands_after_terminal"],
            )
            for r in reports
        ],
        final_owner=session.router.gate.owner,
        final_stage=session.stage,
        final_unsafe=session.unsafe(),
        fault_playing_sent=referee.fault_playing_sent,
        fault_new_kick_rejected=None
        if referee.fault_probe_future is None or not referee.fault_probe_future.done()
        else not referee.fault_probe_future.result().accepted,
        restart_results=session.restart_results,
        arena=arena,
        blocked_walk_samples=len(blocked),
        blocked_nonzero_walker_commands=0,
        scope="scripted simulation referee publishes actual native GameState ROS messages; native capsule/BodyDSD/navigation and actual raw Action; no real UDP GameController or hardware",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arena-manifest", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        choices=[
            "goal_restart",
            "out_restart",
            "pause_walk",
            "stale",
            "penalty",
            "opponent_restart",
            "finished",
            "cancel_preparation",
            "cancel_kick",
            "bad_restart",
            "no_gamestate",
            "manual_stop",
        ],
        default=None,
    )
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--seconds", type=float, default=65.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if (
        os.environ.get("ROS_DOMAIN_ID") != "183"
        or os.environ.get("RMW_IMPLEMENTATION") != "rmw_fastrtps_cpp"
    ):
        raise RuntimeError("Requires explicit isolated Fast DDS domain183")
    args.output.mkdir(parents=True, exist_ok=True)
    GameFlowSession.manifest_path = args.arena_manifest.resolve()
    manifest = json.loads(args.arena_manifest.read_text())
    GameFlowSession.navigation_overrides = {
        "map.ball_diameter": 2 * manifest["ball"]["radius"]
    }
    nav = yaml.safe_load((HERE / "autonomy.yaml").read_text())["navigation"]
    nav.update(GameFlowSession.navigation_overrides)
    nav["use_sim_time"] = True
    params = args.output / "navigation_parameters.yaml"
    params.write_text(
        yaml.safe_dump(
            {"/r1_adapter_test/bitbots_path_planning": {"ros__parameters": nav}}
        )
    )
    rclpy.init(
        args=[
            "--ros-args",
            "--params-file",
            str(params.resolve()),
            "-r",
            "bitbots_path_planning:__ns:=/r1_adapter_test",
        ]
    )
    session = GameFlowSession(args.output, record=args.record)
    if args.scenario == "out_restart":
        session.worker_options["initial_referee_ball_velocity_field"] = [0, 2, 0]
    try:
        session.setup()
        referee = None if args.scenario is None else SimReferee(session, args.scenario)
        session.driver = referee
        session.realtime = args.scenario is None
        while session.seconds < args.seconds:
            session.advance()
            if referee is not None and referee.done(session):
                break
            if session.router.gate.phase == Phase.FAULT and args.scenario not in (
                "cancel_preparation",
                "cancel_kick",
            ):
                break
        arena = session.plant.rpc(dict(op="arena_status"))
        result = (
            verify(session, referee, arena)
            if referee is not None
            else dict(
                mode="external_gamestate",
                seconds=session.seconds,
                game_state_received=session.capsule.received_gamestate(),
                final_owner=session.router.gate.owner,
                final_stage=session.stage,
                final_unsafe=session.unsafe(),
                policy_commands=len(session.kick_commands),
                arena=arena,
                restart_results=session.restart_results,
                scope="real-time simulation consuming native GameState ROS input; no in-process referee publisher; real UDP receiver and hardware not launched",
            )
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        if referee is not None:
            (args.output / "referee_messages.json").write_text(
                json.dumps(referee.messages)
            )
        print(json.dumps(result), flush=True)
    finally:
        if session.driver is not None:
            (args.output / "referee_messages.json").write_text(
                json.dumps(session.driver.messages)
            )
        if session.driver is not None and session.driver.fault_probe is not None:
            session.driver.fault_probe.destroy()
        session.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
