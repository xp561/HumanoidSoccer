"""Unchanged native behavior, instrumented worker and prepared-state capture."""

import argparse
import json
from dataclasses import replace, asdict
from pathlib import Path
import sys

from .autonomy_session import main as autonomy_main
from .match_session import ArenaSession


class CaptureSession(ArenaSession):
    strict_preparation = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker_module = "exp.r1_bitbots_adapter.kick_diagnostic_worker"
        self.worker_options["diagnostic_output"] = str(
            (self.output / "diagnostics").resolve()
        )
        strict = replace(
            self.config,
            pose_error=0.05,
            joint_speed=0.10,
            angular_speed=0.10,
            linear_speed=0.03,
            dwell=0.8,
        )
        (self.output / "capture_setup.json").write_text(
            json.dumps(
                dict(
                    strict_preparation=self.strict_preparation,
                    default_readiness=asdict(self.config),
                    preparation_readiness=asdict(
                        strict if self.strict_preparation else self.config
                    ),
                ),
                indent=2,
            )
        )

    def wait_ready(self, *, preparation, timeout, interruptible=False):
        original = self.config
        if preparation and self.strict_preparation:
            self.config = replace(
                original,
                pose_error=0.05,
                joint_speed=0.10,
                angular_speed=0.10,
                linear_speed=0.03,
                dwell=0.8,
            )
        try:
            return super().wait_ready(
                preparation=preparation, timeout=timeout, interruptible=interruptible
            )
        finally:
            self.config = original


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arena-manifest", type=Path, required=True)
    parser.add_argument("--strict-preparation", action="store_true")
    args, remaining = parser.parse_known_args()
    CaptureSession.strict_preparation = args.strict_preparation
    CaptureSession.manifest_path = args.arena_manifest.resolve()
    manifest = json.loads(CaptureSession.manifest_path.read_text())
    CaptureSession.navigation_overrides = {
        "map.ball_diameter": 2 * manifest["ball"]["radius"]
    }
    sys.argv[1:] = remaining
    autonomy_main(session_class=CaptureSession)


if __name__ == "__main__":
    main()
