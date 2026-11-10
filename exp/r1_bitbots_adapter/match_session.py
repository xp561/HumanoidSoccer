"""Native single-R1 autonomous smoke run in the separately generated arena."""

import argparse
import json
from pathlib import Path
import sys

from .autonomy_session import AutonomousSession, main as autonomy_main
from .match_geometry import bitbots_parameters
from .match_arena import sha256


class ArenaSession(AutonomousSession):
    walker_domain = 183

    def __init__(self, *args, **kwargs):
        self.arena_manifest = json.loads(self.manifest_path.read_text())
        self.worker_module = "exp.r1_bitbots_adapter.match_worker"
        self.worker_options = {"arena_manifest": str(self.manifest_path)}
        self.field_overrides = bitbots_parameters(self.arena_manifest["config"])
        super().__init__(*args, **kwargs)
        if abs(self.map_yaw - self.arena_manifest["field_heading_in_odom_rad"]) > 1e-6:
            raise ValueError("Arena and behavior map headings differ")
        self.map_yaw = float(self.arena_manifest["field_heading_in_odom_rad"])

    def close(self):
        try:
            if self.plant is not None:
                arena = self.plant.rpc(dict(op="arena_status"))
                (self.output / "arena_events.json").write_text(
                    json.dumps(arena, indent=2)
                )
                result_path = self.output / "result.json"
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    result["legacy_shot_metrics"] = result.pop("shot_metrics", None)
                    result["arena"] = arena
                    result["arena_manifest_sha256"] = sha256(self.manifest_path)
                    result["field_parameters"] = self.field_overrides
                    result["scope"] = (
                        "native BodyDSD/navigation single-cycle smoke in provisional Small arena; legacy metrics are not match goals"
                    )
                    result_path.write_text(json.dumps(result, indent=2))
        finally:
            super().close()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arena-manifest", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    ArenaSession.manifest_path = args.arena_manifest.resolve()
    manifest = json.loads(ArenaSession.manifest_path.read_text())
    ArenaSession.navigation_overrides = {
        "map.ball_diameter": 2 * manifest["ball"]["radius"]
    }
    sys.argv[1:] = remainder
    autonomy_main(session_class=ArenaSession)


if __name__ == "__main__":
    main()
