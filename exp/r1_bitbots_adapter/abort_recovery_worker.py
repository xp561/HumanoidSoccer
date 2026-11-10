"""Opt-in cancellation snapshot and paired physical recovery interventions."""

import json
from pathlib import Path

import mujoco
import numpy as np

from . import match_flow_worker
from .match_arena import sha256


class RecoveryProbePlant(match_flow_worker.MatchFlowPlant):
    def __init__(self, request):
        super().__init__(request)
        self.probe_output = Path(request["recovery_probe_output"])
        self.probe_output.mkdir(parents=True, exist_ok=True)
        self.probe_policy_started = False
        self.probe_captured = False

    def goal(self, goal):
        self.probe_policy_started = True
        super().goal(goal)

    def plan(self, request):
        if (
            self.probe_policy_started
            and not self.policy_active
            and not self.probe_captured
        ):
            self.probe_captured = True
            r = self.robot
            spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
            state = np.empty(mujoco.mj_stateSize(r.model, spec))
            mujoco.mj_getState(r.model, r.data, state, spec)
            path = self.probe_output / "cancel_state.npz"
            np.savez(
                path,
                spec=spec,
                integration_state=state,
                actuator_gainprm=r.model.actuator_gainprm.copy(),
                actuator_biasprm=r.model.actuator_biasprm.copy(),
                actuator_forcerange=r.model.actuator_forcerange.copy(),
                positions=self.balance.positions,
                rotations=self.balance.rotations,
                height=self.balance.height,
                orientation=self.balance.orientation,
                posture=request["posture"],
                qpos=r.data.qpos,
                qvel=r.data.qvel,
            )
            (self.probe_output / "cancel_state.json").write_text(
                json.dumps(
                    dict(
                        snapshot_sha256=sha256(path),
                        diagnostics=self.diagnostics(),
                        physics_seconds=float(r.data.time),
                        capture_requested=request.get("capture"),
                        model_sha256=self.manifest["generated_model_sha256"],
                        arena_manifest=str(self.manifest_path),
                        mujoco=mujoco.__version__,
                    ),
                    indent=2,
                )
            )
        return super().plan(request)


def main():
    match_flow_worker.MatchFlowPlant = RecoveryProbePlant
    match_flow_worker.main()


if __name__ == "__main__":
    main()
