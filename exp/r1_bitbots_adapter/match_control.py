"""Conservative single-robot engineering gate over native GameState fields."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MatchPermission:
    allowed: bool
    reason: str
    external_block: bool


class MatchControl:
    def __init__(self, *, team_id=42, max_age=1.0):
        self.team_id = team_id
        self.max_age_ns = round(max_age * 1_000_000_000)
        self.stamp = None
        self.signature = None
        self.invalid = None
        self.terminal = None
        self.terminal_stamp = None
        self.set_seen = False
        self.restart_stamp = None
        self.epoch = 0

    @staticmethod
    def time(msg):
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    @staticmethod
    def fields(msg):
        return (
            msg.main_state,
            msg.game_phase,
            msg.stopped,
            msg.penalized,
            msg.penalized_in_place,
            msg.has_red_card,
            msg.penalty,
            msg.set_play,
            msg.kicking_team,
        )

    def ingest(self, msg, now_ns):
        stamp, signature = self.time(msg), self.fields(msg)
        if stamp <= 0 or stamp > now_ns or now_ns - stamp > self.max_age_ns:
            self.invalid = "invalid_gamestate_timestamp"
            return
        if self.stamp is not None and stamp < self.stamp:
            self.invalid = "out_of_order_gamestate"
            return
        if stamp == self.stamp and signature != self.signature:
            self.invalid = "conflicting_gamestate_timestamp"
            return
        self.stamp, self.signature, self.invalid = stamp, signature, None
        if (
            self.terminal is not None
            and msg.main_state == msg.STATE_SET
            and stamp > self.terminal_stamp
        ):
            self.set_seen = True

    def permission(self, msg, now_ns):
        if self.invalid:
            return MatchPermission(False, self.invalid, True)
        if self.stamp is None:
            return MatchPermission(False, "no_gamestate", True)
        if not 0 <= now_ns - self.stamp <= self.max_age_ns:
            return MatchPermission(False, "stale_gamestate", True)
        if self.time(msg) != self.stamp or self.fields(msg) != self.signature:
            return MatchPermission(False, "native_capsule_not_synchronized", True)
        if msg.main_state != msg.STATE_PLAYING:
            return MatchPermission(False, "main_state_not_playing", True)
        if msg.game_phase != msg.GAME_PHASE_NORMAL:
            return MatchPermission(False, "unsupported_game_phase", True)
        if msg.stopped:
            return MatchPermission(False, "game_controller_stopped", True)
        if (
            msg.penalized
            or msg.penalized_in_place
            or msg.penalty != msg.PENALTY_NONE
            or msg.has_red_card
        ):
            return MatchPermission(False, "player_penalized", True)
        if msg.set_play != msg.SET_PLAY_NONE and msg.kicking_team != self.team_id:
            return MatchPermission(False, "opponent_set_play_hold", True)
        if self.terminal is not None:
            return MatchPermission(False, "arena_terminal_waiting_restart", False)
        if self.restart_stamp is not None and self.stamp <= self.restart_stamp:
            return MatchPermission(False, "waiting_new_playing_after_restart", True)
        return MatchPermission(True, "playing", False)

    def latch_terminal(self, event, now_ns):
        if self.terminal is None:
            self.terminal, self.terminal_stamp = dict(event), now_ns
            self.set_seen = False

    def restart_state_valid(self, msg, now_ns):
        return bool(
            self.terminal is not None
            and self.set_seen
            and self.invalid is None
            and self.stamp is not None
            and 0 <= now_ns - self.stamp <= self.max_age_ns
            and self.time(msg) == self.stamp
            and self.fields(msg) == self.signature
            and msg.main_state == msg.STATE_SET
            and msg.stopped
        )

    def acknowledge_restart(self):
        if self.terminal is None or not self.set_seen:
            raise RuntimeError("Restart lacks a latched terminal and SET sequence")
        self.terminal = self.terminal_stamp = None
        self.restart_stamp = self.stamp
        self.set_seen = False
        self.epoch += 1
