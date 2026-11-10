"""Native GameState capsule controls an opt-in, continuous single-R1 arena."""

import json

from std_srvs.srv import Trigger

from .autonomy_session import cancel_abandoned_kick
from .capture_timed_kick import TimedCaptureSession
from .handoff import Phase
from .match_control import MatchControl
from .mujoco_bridge import state_arrays
from .readiness import Readiness


class GameFlowSession(TimedCaptureSession):
    def __init__(self, *args, **kwargs):
        self.match_control = MatchControl()
        self.match_transitions = []
        self.flow_trace = []
        self.restart_results = []
        self.driver = None
        self.last_permission = None
        self.last_native_frame = None
        self.walk_command_permission = (None, False)
        self.match_feed = dict(epoch=0, events=[])
        self.stable_monitor = None
        self.last_stable_ns = None
        super().__init__(*args, **kwargs)
        self.worker_module = "exp.r1_bitbots_adapter.match_flow_worker"
        self.worker_options.pop("diagnostic_output", None)

    def configure_recovery(self):
        super().configure_recovery()
        if self.outcome == "cancelled":
            # The robot moved since preparation. Re-anchor the existing balance
            # planner at the measured terminal pose, without changing live state.
            # This improves supported stops; airborne cancellation can still fault.
            self.posture = self.target.copy()
            self.capture_balance = True

    @property
    def capsule(self):
        return self.body.dsd.blackboard.gamestate

    def permission(self):
        msg = self.capsule.gamestate
        frame = (self.match_control.time(msg), self.match_control.fields(msg))
        if frame != self.last_native_frame:
            self.match_control.ingest(msg, self.ns)
            self.last_native_frame = frame
        return self.match_control.permission(msg, self.ns)

    def motion_permitted(self):
        return bool(
            self.permission().allowed
            and not getattr(self, "manual_stop_requested", False)
        )

    def setup(self, **kwargs):
        super().setup(**kwargs)
        original = self.walker.publish_target

        def match_fenced_target(ns, *, stopping=False):
            allowed = self.motion_permitted()
            self.walk_command_permission = (ns, allowed)
            return original(ns, stopping=stopping or not allowed)

        self.walker.publish_target = match_fenced_target
        self.restart_server = self.bridge.create_service(
            Trigger, "match/restart", self.restart_request
        )
        self.restart_client = self.bridge.create_client(Trigger, "match/restart")
        self.stable_monitor = Readiness(self.config, self.nominal_height)

    def before_body_tick(self):
        permission = self.permission()
        manual = getattr(self, "manual_stop_requested", False)
        kick_type = self.body.dsd.actions["R1Kick"]
        current_action_active = any(
            isinstance(instance, kick_type) for _, instance in self.body.dsd.stack
        )
        transaction_clear = (
            self.router.gate.phase == Phase.READY
            and not self.supervisor.reserved
            and not self.adapter.busy
        )
        self.context.enabled = (
            permission.allowed
            and not manual
            and (transaction_clear or current_action_active)
        )
        reason = "manual_stop" if manual else permission.reason
        external_block = permission.external_block or manual
        if reason != self.last_permission:
            self.match_transitions.append(
                dict(
                    seconds=self.seconds,
                    reason=reason,
                    motion_allowed=self.context.enabled,
                    stage=self.stage,
                    owner=self.router.gate.owner,
                )
            )
        if external_block:
            self.cancel.set()
            if reason != self.last_permission:
                self.body.dsd.interrupt()
            cancel_abandoned_kick(self.body.dsd)
            self.body.dsd.blackboard.pathfinding.stop_walk()
        # Local goal/out inhibits new work and navigation immediately. An active
        # kick drains through its validated handoff; external STOP cancels it.
        self.last_permission = reason
        if self.stable_monitor is not None and self.last_stable_ns != self.ns:
            self.last_stable_ns = self.ns
            if self.router.gate.phase == Phase.READY and not self.motion_permitted():
                self.stable_monitor.update(self.seconds, self.measured)
            else:
                self.stable_monitor = Readiness(self.config, self.nominal_height)

    def sync(self, *, owner):
        if self.driver is not None and hasattr(self, "body"):
            self.driver.tick(self)
        return super().sync(owner=owner)

    def consume(self, command):
        command_owner, command_ns = self.router.gate.owner, self.ns
        result = super().consume(command)
        if "match_feed" in result:
            self.match_feed = result["match_feed"]
            for event in self.match_feed["events"]:
                self.match_control.latch_terminal(event, self.ns)
        if hasattr(self, "body"):
            p = self.permission()
            self.flow_trace.append(
                dict(
                    seconds=self.seconds,
                    stage=self.stage,
                    owner=self.router.gate.owner,
                    motion_allowed=p.allowed
                    and not getattr(self, "manual_stop_requested", False)
                    and self.router.gate.phase != Phase.FAULT,
                    game_state_allows_motion=p.allowed,
                    behavior_enabled=self.context.enabled,
                    command_owner=command_owner,
                    command_ns=command_ns,
                    walk_permitted_at_command=self.walk_command_permission[1]
                    if self.walk_command_permission[0] == command_ns
                    else None,
                    reason=p.reason,
                    main_state=self.capsule.get_main_state(),
                    native_stopped=self.capsule.is_stopped(),
                    native_penalized=self.capsule.get_is_penalized(),
                    kick_commands=len(self.kick_commands),
                    epoch=self.match_control.epoch,
                    root_pos=self.state["root_pos"].tolist(),
                    measured=self.measured,
                    walker_requested_command=self.walker.command.tolist()
                    if hasattr(self, "walker")
                    else None,
                    walker_applied_command=getattr(
                        self.walker, "last_applied_command", None
                    ).tolist()
                    if hasattr(self, "walker")
                    and hasattr(self.walker, "last_applied_command")
                    else None,
                )
            )
        return result

    def restart_request(self, request, response):
        self.permission()
        valid = self.match_control.restart_state_valid(self.capsule.gamestate, self.ns)
        valid = bool(
            valid
            and self.router.gate.phase == Phase.READY
            and not self.adapter.busy
            and not self.supervisor.reserved
            and not getattr(self, "manual_stop_requested", False)
            and self.stable_monitor is not None
            and self.stable_monitor.elapsed >= self.config.dwell
        )
        if not valid:
            response.success = False
            response.message = "Restart requires fresh stopped SET, a terminal epoch, and sustained stable READY"
        else:
            try:
                reply = self.plant.rpc(
                    dict(
                        op="restart_ball",
                        stopped_ack=True,
                        ball_field=[1.35, 0.0, self.arena_manifest["ball"]["radius"]],
                    )
                )
                self.state = state_arrays(reply)
                self.measured = reply["diagnostics"]
                self.match_feed = reply["match_feed"]
                self.match_control.acknowledge_restart()
                assert self.match_control.epoch == self.match_feed["epoch"]
                response.success = True
                response.message = "Explicit referee ball placement completed; waiting for newer PLAYING"
            except (ValueError, RuntimeError) as error:
                response.success = False
                response.message = str(error)
        self.restart_results.append(
            dict(
                seconds=self.seconds,
                success=response.success,
                message=response.message,
                epoch=self.match_control.epoch,
            )
        )
        return response

    def close(self):
        self.driver = None
        try:
            if self.plant is not None:
                (self.output / "match_transitions.json").write_text(
                    json.dumps(self.match_transitions, indent=2)
                )
                (self.output / "match_trace.json").write_text(
                    json.dumps(self.flow_trace)
                )
                (self.output / "restart_results.json").write_text(
                    json.dumps(self.restart_results, indent=2)
                )
        finally:
            super().close()
