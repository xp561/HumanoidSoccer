"""Explicit motion ownership; readiness acknowledgements are supplied externally."""
from enum import Enum
import math


class Phase(str, Enum):
    READY = "ready"
    STOPPING = "stopping"
    PREPARING = "preparing"
    KICKING = "kicking"
    RECOVERING = "recovering"
    FAULT = "fault"


class Handoff:
    def __init__(self, *, walker_available=False, phase_timeout=10., command_max_age=.1):
        if not math.isfinite(phase_timeout) or phase_timeout <= 0:
            raise ValueError("Invalid phase timeout")
        if not math.isfinite(command_max_age) or command_max_age <= 0:
            raise ValueError("Invalid command age")
        self.walker_available = walker_available
        self.phase_timeout = phase_timeout
        self.command_max_age_ns = round(command_max_age*1e9)
        self.phase = Phase.READY
        self.generation = 0
        self.entered_ns = 0
        self.last_now_ns = 0
        self.reason = None
        self.outcome = None
        self.transitions = []

    @property
    def owner(self):
        if self.phase in (Phase.READY, Phase.STOPPING):
            return "walk" if self.walker_available else "hold"
        return {Phase.PREPARING:"prepare", Phase.KICKING:"kick",
                Phase.RECOVERING:"recover", Phase.FAULT:"hcm"}[self.phase]

    def _time(self, now_ns):
        if not isinstance(now_ns, int) or now_ns <= 0:
            raise ValueError("Expected a positive integer timestamp")
        if now_ns < self.last_now_ns:
            self.phase = Phase.FAULT
            self.reason = "clock_regression"
            self.generation += 1
            raise ValueError("Clock moved backwards")
        self.last_now_ns = now_ns

    def _transition(self, phase, now_ns):
        self._time(now_ns)
        self.phase = phase
        self.entered_ns = now_ns
        self.transitions.append(dict(phase=phase.value, owner=self.owner,
                                     generation=self.generation, stamp_ns=now_ns))

    def begin(self, now_ns):
        self._time(now_ns)
        if self.phase != Phase.READY:
            raise RuntimeError("A motion transaction is already active or faulted")
        self.generation += 1
        self.reason = self.outcome = None
        self._transition(Phase.STOPPING, now_ns)
        return self.generation

    def acknowledge(self, expected_phase, generation, now_ns, *, success=True, outcome=None):
        self.tick(now_ns)
        if generation != self.generation or self.phase != expected_phase:
            return False
        if not success:
            self.fail("readiness_failed:"+self.phase.value, now_ns)
            return True
        next_phase = {Phase.STOPPING:Phase.PREPARING,
                      Phase.PREPARING:Phase.KICKING,
                      Phase.KICKING:Phase.RECOVERING,
                      Phase.RECOVERING:Phase.READY}
        if expected_phase not in next_phase:
            raise ValueError("This phase does not accept an ordinary acknowledgement")
        if expected_phase == Phase.KICKING:
            self.outcome = outcome
        self._transition(next_phase[expected_phase], now_ns)
        return True

    def fail(self, reason, now_ns):
        self.reason = reason
        self.generation += 1
        self._transition(Phase.FAULT, now_ns)

    def recover_fault(self, now_ns):
        self._time(now_ns)
        if self.phase != Phase.FAULT:
            raise RuntimeError("No fault to recover")
        self.generation += 1
        self._transition(Phase.RECOVERING, now_ns)
        return self.generation

    def tick(self, now_ns):
        self._time(now_ns)
        if self.phase not in (Phase.READY, Phase.FAULT) and now_ns-self.entered_ns > round(self.phase_timeout*1e9):
            self.fail("phase_timeout:"+self.phase.value, now_ns)

    def allows(self, source, stamp_ns, now_ns):
        self.tick(now_ns)
        if source != self.owner or not isinstance(stamp_ns, int):
            return False
        # The ownership fence discards delayed packets from an earlier phase.
        return (stamp_ns > 0 and self.entered_ns <= stamp_ns <= now_ns
                and now_ns-stamp_ns <= self.command_max_age_ns)
