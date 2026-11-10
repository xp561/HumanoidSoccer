"""One real ROS kick plus original measured recovery from a shared snapshot.

Each replay is a fresh simulated world initialized once. Navigation and
preparation are intentionally excluded from the paired intervention.
"""

import argparse
import json
import os
from pathlib import Path

import rclpy

from .walking_session import WalkingSession
from .handoff import Phase


class PreparedSession(WalkingSession):
    walker_domain = 185

    def __init__(self, output, manifest, snapshot):
        self.worker_module = "exp.r1_bitbots_adapter.kick_diagnostic_worker"
        self.worker_options = dict(
            arena_manifest=str(manifest.resolve()),
            diagnostic_output=str((output / "diagnostics").resolve()),
            snapshot_path=str(snapshot.resolve()),
        )
        super().__init__(output, realtime=False, render=False)

    def run_prepared(self):
        self.events, self.phase_trace = [], []
        self.shot_kick_start = len(self.kick_commands)
        self.terminal_kick_count = None
        self.outcome = None
        gate = self.router.gate
        generation = gate.begin(self.ns)
        # Snapshot receipt supplies the already-prepared state. This bypass is
        # exclusive to this diagnostic entry, not admission in ordinary behavior.
        assert gate.acknowledge(Phase.STOPPING, generation, self.ns)
        assert gate.acknowledge(Phase.PREPARING, generation, self.ns)
        self.event("kicking", "paired fresh-world replay of prepared-state snapshot")
        failure = self.execute_kick()
        self.outcome = failure or "policy_completed"
        if gate.phase == Phase.FAULT:
            self.fault(gate.reason)
            self.save_report(False)
            return
        assert gate.acknowledge(
            Phase.KICKING, generation, self.ns, outcome=self.outcome
        )
        self.target = self.state["joint_pos"][self.adapter.runtime.ids].copy()
        self.posture = (
            self.target.copy()
            if self.outcome == "policy_completed"
            else self.nominal.copy()
        )
        self.capture_balance = self.outcome == "policy_completed"
        self.plant.rpc(dict(op="end_policy"))
        self.event(
            "recovering", "original balance feedback and sustained support check"
        )
        failure = self.wait_ready(
            preparation=False, timeout=self.config.recovery_timeout
        )
        if failure is not None:
            self.fault(failure)
            self.save_report(False)
            return
        assert gate.acknowledge(Phase.RECOVERING, generation, self.ns)
        self.event("ready", "paired replay reached measured stable recovery")
        self.save_report(True)

    def close(self):
        try:
            if self.plant is not None:
                arena = self.plant.rpc(dict(op="arena_status"))
                (self.output / "arena_events.json").write_text(
                    json.dumps(arena, indent=2)
                )
        finally:
            if self.plant is not None:
                self.plant.close()
            if hasattr(self, "executor"):
                self.executor.shutdown(timeout_sec=3)
                for name in ("walker", "bridge", "router", "adapter"):
                    if hasattr(self, name):
                        getattr(self, name).destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arena-manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") != "185":
        raise RuntimeError("Prepared replay requires isolated simulation domain 185")
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    session = PreparedSession(args.output, args.arena_manifest, args.snapshot)
    try:
        session.setup()
        session.run_prepared()
        report = json.loads(Path(session.last_report).read_text())
        summary = dict(
            recovered=report["recovered"],
            outcome=report["outcome"],
            final_stage=report["final_stage"],
            final_owner=report["final_owner"],
            policy_commands=report["policy_commands"],
            late_commands=report["policy_commands_after_terminal"],
            source_snapshot=str(args.snapshot.resolve()),
            arena_manifest=str(args.arena_manifest.resolve()),
            final_root_pos=session.state["root_pos"].tolist(),
            final_ball_pos=session.state["ball_pos"].tolist(),
            actual_goal=session.goal,
            scope="paired diagnostic policy+recovery; no new navigation/preparation; initial snapshot write only",
        )
        (args.output / "result.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        session.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
