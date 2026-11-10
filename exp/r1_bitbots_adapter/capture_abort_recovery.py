"""Capture the actual native GameState STOP/raw Action cancellation state."""

import argparse
import sys

from . import validate_match_flow


BaseSession = validate_match_flow.GameFlowSession


class CaptureAbortSession(BaseSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker_module = "exp.r1_bitbots_adapter.abort_recovery_worker"
        self.worker_options["recovery_probe_output"] = str(self.output / "snapshot")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cancel-after-commands", type=int, default=30)
    args, remaining = parser.parse_known_args()
    if not 1 <= args.cancel_after_commands <= 259:
        parser.error("cancellation must precede the 260-command policy end")
    validate_match_flow.SimReferee.cancel_after_commands = args.cancel_after_commands
    sys.argv[1:] = remaining
    validate_match_flow.GameFlowSession = CaptureAbortSession
    validate_match_flow.main()


if __name__ == "__main__":
    main()
