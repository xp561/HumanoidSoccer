"""Opt-in continuous walker/kicker session; legacy live session remains available."""

import json
from pathlib import Path
from rclpy.parameter import Parameter
from .live_session import Session
from .handoff import Phase
from .mujoco_bridge import spin_until
from .walking_ros import WalkingNode


class WalkingSession(Session):
    enable_walking = True

    def run(self):
        try:
            super().run()
        finally:
            if hasattr(self, "walker"):
                self.walker.destroy_node()

    def setup(self, **kwargs):
        super().setup(**kwargs)
        self.walker = WalkingNode(
            isolated_domain=getattr(self, "walker_domain", 179),
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.executor.add_node(self.walker)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.router.gate.walker_available = True
        if hasattr(self, "walker"):
            self.walker.runtime.reset_history()
            self.walker.command[:] = 0
            self.walker.command_ns = None
        self.was_walking = False

    def controlled_step(self, source, target):
        if source != "walk":
            self.was_walking = False
            return super().controlled_step(source, target)
        self.sync(owner=False)
        spin_until(self.executor, lambda: self.walker.ready(self.ns))
        if not self.was_walking:
            self.walker.runtime.reset_history()
        before = len(self.bridge.commands)
        self.walker.publish_target(
            self.ns, stopping=self.router.gate.phase == Phase.STOPPING
        )
        spin_until(self.executor, lambda: len(self.bridge.commands) > before)
        if len(self.bridge.commands) != before + 1:
            raise RuntimeError("Competing walker motor commands")
        self.was_walking = True
        reply = self.consume(self.bridge.commands[-1])
        # Session's posture/target arrays always use kicking action order.
        self.target = self.state["joint_pos"][self.adapter.runtime.ids].copy()
        if self.router.gate.phase == Phase.STOPPING:
            self.posture = self.target.copy()
        return reply

    def update_view(self, **kwargs):
        super().update_view(**kwargs)
        with self.lock:
            self.status["scope"] = "continuous single-robot walk/kick simulation"
            self.status["walking_available"] = True

    def event(self, stage, reason):
        if stage == "ready":
            reason = "measured stable recovery; walking ownership restored"
        super().event(stage, reason)

    def save_report(self, recovered):
        super().save_report(recovered)
        path = Path(self.last_report)
        report = json.loads(path.read_text())
        report["scope"] = (
            "continuous same-model walker/kicker simulation; no hardware or get-up"
        )
        report["walking_contract"] = str(
            self.walker.runtime.contract["artifact_version"]
        )
        path.write_text(json.dumps(report, indent=2))


def main():
    import argparse
    import rclpy
    import signal
    from .live_web import PAGE, serve

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("logs/r1_logic_fix/r1_walking_live")
    )
    parser.add_argument("--port", type=int, default=8878)
    args = parser.parse_args()
    session = WalkingSession(args.output)
    session.page = (
        PAGE.decode()
        .replace(
            "READY 只表示已测得稳定并交还保持控制器，尚未接入行走或倒地起身。",
            "READY 表示稳定恢复后交还行走控制器。行走接受 cmd_vel；当前为单机仿真。",
        )
        .encode()
    )

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    rclpy.init()
    try:
        serve(session, args.port)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
