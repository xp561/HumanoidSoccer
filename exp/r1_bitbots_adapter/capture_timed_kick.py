"""Native arena cycle with an explicit diagnostic raw-policy duration."""

import argparse
import json
import os
from pathlib import Path
import sys

from .autonomy_session import main as autonomy_main
from .capture_kick_session import CaptureSession


class TimedCaptureSession(CaptureSession):
    execution_seconds = 5.2

    def setup(self, **kwargs):
        super().setup(**kwargs)
        self.adapter.cfg["execution_seconds"] = self.execution_seconds
        (self.output / "timing_setup.json").write_text(
            json.dumps(
                dict(
                    execution_seconds=self.execution_seconds,
                    rmw_implementation=os.environ["RMW_IMPLEMENTATION"],
                    scope="opt-in diagnostic duration; unchanged native behavior, preparation gates, frozen policies and recovery",
                ),
                indent=2,
            )
        )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arena-manifest", type=Path, required=True)
    parser.add_argument("--duration", type=float, choices=[5.2, 6.0, 8.0], required=True)
    args, remaining = parser.parse_known_args()
    TimedCaptureSession.execution_seconds = args.duration
    TimedCaptureSession.manifest_path = args.arena_manifest.resolve()
    manifest = json.loads(TimedCaptureSession.manifest_path.read_text())
    TimedCaptureSession.navigation_overrides = {"map.ball_diameter": 2 * manifest["ball"]["radius"]}
    sys.argv[1:] = remaining
    if os.environ.get("RMW_IMPLEMENTATION") != "rmw_fastrtps_cpp":
        raise RuntimeError("Native diagnostic requires explicit Fast DDS inside Pixi")
    autonomy_main(session_class=TimedCaptureSession)


if __name__ == "__main__":
    main()
