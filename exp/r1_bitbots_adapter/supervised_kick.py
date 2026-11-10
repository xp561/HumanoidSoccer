"""Simulation action spanning measured stopping, raw skill, and recovery."""

import math
import os
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.task import Future
from bitbots_msgs.action import Kick
from .handoff import Phase


class SupervisedKick(Node):
    def __init__(self, session, *, isolated_domain=181, **kwargs):
        super().__init__("r1_supervised_kick", namespace="/r1_adapter_test", **kwargs)
        if isolated_domain not in (181, 183) or os.environ.get("ROS_DOMAIN_ID") != str(
            isolated_domain
        ):
            raise RuntimeError(
                "Autonomy supervisor requires isolated simulation domain"
            )
        self.session = session
        self.reserved = False
        self.pending = None
        self.done = None
        self.server = ActionServer(
            self,
            Kick,
            "supervised_kick",
            self.execute,
            goal_callback=self.goal,
            cancel_callback=self.cancel,
            callback_group=ReentrantCallbackGroup(),
        )
        self.results = []

    def goal(self, request):
        permitted = getattr(self.session, "motion_permitted", None)
        if permitted is not None and not permitted():
            return GoalResponse.REJECT
        values = [request.x, request.y, request.strength, request.timeout]
        if (
            self.reserved
            or self.session.router.gate.phase != Phase.READY
            or not all(math.isfinite(v) for v in values)
            or request.strength != 0
            or request.timeout > 0
            or math.hypot(request.x, request.y) < 1e-12
            or abs(math.atan2(request.y, request.x))
            > self.session.adapter.cfg["heading_tolerance"]
        ):
            return GoalResponse.REJECT
        self.reserved = True
        self.session.cancel.clear()
        return GoalResponse.ACCEPT

    def cancel(self, handle):
        self.session.cancel.set()
        return CancelResponse.ACCEPT

    async def execute(self, handle):
        self.done = Future()
        self.pending = handle
        try:
            report = await self.done
            cancelled = handle.is_cancel_requested
            success = (
                not cancelled
                and report["recovered"]
                and report["outcome"] == "policy_completed"
            )
            result = Kick.Result()
            if cancelled:
                handle.canceled()
                result.result = Kick.Result.ABORTED
            elif success:
                handle.succeed()
                result.result = Kick.Result.SUCCESS
            else:
                handle.abort()
                result.result = Kick.Result.ABORTED
            self.results.append(
                dict(
                    success=success,
                    cancelled=cancelled,
                    recovered=report["recovered"],
                    outcome=report["outcome"],
                )
            )
            return result
        finally:
            self.reserved = False
            self.pending = None
            self.done = None

    def complete(self, report):
        if self.done is None or self.done.done():
            raise RuntimeError("No active supervised transaction")
        self.done.set_result(report)
