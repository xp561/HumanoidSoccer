"""State-machine checks independent of ROS and robot availability."""
import unittest
from .handoff import Handoff, Phase


class TestHandoff(unittest.TestCase):
    def setUp(self):
        self.gate=Handoff()
        self.now=1_000_000_000

    def advance(self,phase,token,success=True,**kwargs):
        self.now+=20_000_000
        return self.gate.acknowledge(phase,token,self.now,success=success,**kwargs)

    def kicking(self):
        token=self.gate.begin(self.now)
        self.advance(Phase.STOPPING,token)
        self.advance(Phase.PREPARING,token)
        return token

    def test_no_walking_policy_means_hold_owner(self):
        self.assertEqual(self.gate.owner,"hold")
        self.assertFalse(self.gate.allows("walk",self.now,self.now))

    def test_normal_transaction_returns_to_hold(self):
        token=self.kicking()
        self.assertEqual(self.gate.owner,"kick")
        self.advance(Phase.KICKING,token,outcome="success")
        self.assertEqual(self.gate.owner,"recover")
        self.advance(Phase.RECOVERING,token)
        self.assertEqual(self.gate.phase,Phase.READY)
        self.assertEqual(self.gate.owner,"hold")
        self.assertEqual(self.gate.outcome,"success")

    def test_abort_requires_recovery_before_new_request(self):
        token=self.kicking()
        self.advance(Phase.KICKING,token,outcome="aborted")
        with self.assertRaises(RuntimeError):
            self.gate.begin(self.now)
        self.advance(Phase.RECOVERING,token)
        self.assertEqual(self.gate.owner,"hold")

    def test_cannot_skip_stop_or_preparation(self):
        token=self.gate.begin(self.now)
        self.assertFalse(self.advance(Phase.PREPARING,token))
        self.assertFalse(self.advance(Phase.KICKING,token))
        self.assertEqual(self.gate.phase,Phase.STOPPING)

    def test_old_generation_ack_does_not_release_fault(self):
        token=self.kicking()
        self.gate.fail("fall",self.now)
        self.assertFalse(self.advance(Phase.KICKING,token))
        self.assertEqual(self.gate.owner,"hcm")

    def test_clock_regression_fails_closed(self):
        self.kicking()
        with self.assertRaises(ValueError):
            self.gate.allows("kick",self.now-1,self.now-1)
        self.assertEqual(self.gate.phase,Phase.FAULT)

    def test_packet_timestamp_fence(self):
        self.kicking()
        self.assertFalse(self.gate.allows("kick",self.now-1,self.now))
        self.assertTrue(self.gate.allows("kick",self.now,self.now))

    def test_future_and_stale_commands_rejected(self):
        self.kicking()
        self.assertFalse(self.gate.allows("kick",self.now+1,self.now))
        self.assertFalse(self.gate.allows("kick",self.now,self.now+200_000_000))

    def test_exactly_one_owner_per_phase(self):
        token=self.gate.begin(self.now)
        for phase in (Phase.STOPPING,Phase.PREPARING,Phase.KICKING,Phase.RECOVERING):
            sources=("hold","walk","prepare","kick","recover","hcm")
            allowed=[s for s in sources if self.gate.allows(s,self.now,self.now)]
            self.assertEqual(allowed,[self.gate.owner])
            self.advance(phase,token)

    def test_preparation_failure_does_not_start_kick(self):
        token=self.gate.begin(self.now)
        self.advance(Phase.STOPPING,token)
        self.advance(Phase.PREPARING,token,success=False)
        self.assertEqual(self.gate.phase,Phase.FAULT)
        self.assertEqual(self.gate.owner,"hcm")

    def test_timeout_requires_explicit_recovery(self):
        self.gate.begin(self.now)
        self.gate.tick(self.now+11_000_000_000)
        self.assertEqual(self.gate.phase,Phase.FAULT)
        with self.assertRaises(RuntimeError):
            self.gate.begin(self.now+11_000_000_000)
        token=self.gate.recover_fault(self.now+11_000_000_000)
        self.now+=11_000_000_000
        self.advance(Phase.RECOVERING,token)
        self.assertEqual(self.gate.owner,"hold")

    def test_optional_walking_owner_remains_during_stop(self):
        gate=Handoff(walker_available=True)
        token=gate.begin(self.now)
        self.assertEqual(gate.owner,"walk")
        gate.acknowledge(Phase.STOPPING,token,self.now)
        self.assertEqual(gate.owner,"prepare")


if __name__=="__main__":
    unittest.main()
