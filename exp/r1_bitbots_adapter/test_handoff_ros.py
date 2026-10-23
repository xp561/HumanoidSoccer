"""Native ROS checks for the real Bit-Bots client and isolated lease router."""
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import threading
import time
import unittest
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from bitbots_msgs.action import Kick
from bitbots_msgs.msg import JointCommand
from . import test_ros as ros_fixture
wait=ros_fixture.wait
from .ownership_ros import OwnershipRouter
from .handoff import Phase

SOURCE=Path(os.environ.get("R1_BITBOTS_SOURCE","/home/tl/Downloads/r1_ros_integration/bitbots_source"))
sys.path[:0]=[str(SOURCE/"src/bitbots_behavior/bitbots_blackboard"),
              str(SOURCE/"src/bitbots_misc/bitbots_utils")]
from bitbots_blackboard.capsules.kick_capsule import KickCapsule


class TestCapsuleNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ros_fixture.TestROS.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        ros_fixture.TestROS.tearDownClass.__func__(cls)

    def setUp(self):
        ros_fixture.TestROS.setUp(self)
        ref=self.adapter.references[0]
        q=ref.data["body_quat_w"][0,ref.root]
        yaw=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
        self.yaw_deg=math.degrees(yaw)
        board=SimpleNamespace(config={"rl_kick":{"timeout":-1.}},
            world_model=SimpleNamespace(get_current_position=lambda:(0.,0.,yaw)))
        self.capsule=KickCapsule(self.fixture,board)

    def test_dsd_interruption_cancels_running_action(self):
        import importlib.util
        from types import ModuleType
        from unittest.mock import patch
        dummy=ModuleType("bitbots_blackboard.body_blackboard")
        dummy.BodyBlackboard=object
        path=SOURCE/"src/bitbots_behavior/bitbots_body_behavior/bitbots_body_behavior/behavior_dsd/actions/kick_ball.py"
        spec=importlib.util.spec_from_file_location("r1_native_dsd_kick_test",path)
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{"bitbots_blackboard.body_blackboard":dummy}):
            spec.loader.exec_module(module)
        board=self.capsule._blackboard
        board.kick=self.capsule
        board.world_model.forget_ball=lambda:None
        action=module.RLKick(board,SimpleNamespace(),{})
        self.assertTrue(self.capsule.start_rl_kick(self.yaw_deg,0.))
        wait(lambda:self.capsule._rl_kick_goal_handle is not None)
        action.on_pop()
        wait(lambda:not self.capsule.is_currently_kicking)
        self.assertEqual(self.capsule.last_rl_kick_status,GoalStatus.STATUS_CANCELED)

    def test_actual_client_completes(self):
        self.assertTrue(self.capsule.start_rl_kick(self.yaw_deg,0.))
        wait(lambda:not self.capsule.is_currently_kicking)
        self.assertEqual(self.capsule.last_rl_kick_status,GoalStatus.STATUS_SUCCEEDED)
        self.assertEqual(self.capsule.last_rl_kick_result,Kick.Result.SUCCESS)

    def test_actual_client_handles_rejection(self):
        self.assertTrue(self.capsule.start_rl_kick(self.yaw_deg,3.))
        wait(lambda:not self.capsule.is_currently_kicking)
        self.assertEqual(self.capsule.last_rl_kick_result,Kick.Result.REJECTED)
        self.assertFalse(self.fixture.commands)

    def test_actual_client_cancel_before_goal_ack(self):
        self.assertTrue(self.capsule.start_rl_kick(self.yaw_deg,0.))
        self.assertTrue(self.capsule.stop_rl_kick())
        wait(lambda:not self.capsule.is_currently_kicking)
        self.assertEqual(self.capsule.last_rl_kick_status,GoalStatus.STATUS_CANCELED)
        self.assertEqual(self.capsule.last_rl_kick_result,Kick.Result.ABORTED)


class TestOwnershipROS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.router=OwnershipRouter()
        cls.peer=Node("ownership_test_inputs",namespace="/r1_adapter_test")
        cls.received=[]
        cls.peer.create_subscription(JointCommand,"leased_motor_goals",cls.received.append,100)
        topics={"walk":"walking_motor_goals","hold":"hold_motor_goals","prepare":"prepare_motor_goals",
                "kick":"kick_motor_goals","recover":"recovery_motor_goals","hcm":"hcm_motor_goals"}
        cls.pubs={source:cls.peer.create_publisher(JointCommand,topic,100) for source,topic in topics.items()}
        cls.executor=SingleThreadedExecutor()
        cls.executor.add_node(cls.router);cls.executor.add_node(cls.peer)
        cls.thread=threading.Thread(target=cls.executor.spin,daemon=True)
        cls.thread.start()
        wait(lambda:all(p.get_subscription_count() for p in cls.pubs.values()))

    @classmethod
    def tearDownClass(cls):
        cls.executor.shutdown(timeout_sec=3)
        cls.thread.join(3)
        cls.peer.destroy_node();cls.router.destroy_node();rclpy.shutdown()

    def setUp(self):
        from .handoff import Handoff
        with self.router.lock:
            self.router.gate=Handoff()
        time.sleep(.03)
        self.router.forwarded.clear();self.router.rejected.clear();self.received.clear()

    def message(self,stamp=None):
        msg=JointCommand()
        msg.header.stamp=self.peer.get_clock().now().to_msg() if stamp is None else stamp
        msg.joint_names=list(self.router.names)
        msg.positions=[0.]*len(msg.joint_names)
        msg.kp=[1.]*len(msg.joint_names);msg.kd=[.1]*len(msg.joint_names)
        msg.max_torques=[1.]*len(msg.joint_names)
        return msg

    def bombard(self):
        start=len(self.router.forwarded)+len(self.router.rejected)
        out=len(self.received)
        for source,pub in self.pubs.items():
            msg=self.message()
            if source=="walk":
                msg.from_hcm=True
            pub.publish(msg)
        wait(lambda:len(self.router.forwarded)+len(self.router.rejected)==start+len(self.pubs))
        wait(lambda:len(self.received)==out+1)
        self.assertEqual(self.router.forwarded[-1]["source"],self.router.gate.owner)

    def test_all_phases_have_one_output_owner(self):
        self.bombard()
        token=self.router.begin_transaction()
        for phase in (Phase.STOPPING,Phase.PREPARING,Phase.KICKING,Phase.RECOVERING):
            self.bombard()
            self.assertTrue(self.router.acknowledge_phase(phase,token))
        self.bombard()
        self.assertEqual(self.router.gate.owner,"hold")
        self.assertEqual(len(self.router.forwarded),6)
        self.assertEqual(len(self.router.rejected),30)

    def test_delayed_previous_phase_packet_dropped(self):
        old=self.message()
        token=self.router.begin_transaction()
        self.router.acknowledge_phase(Phase.STOPPING,token)
        self.router.acknowledge_phase(Phase.PREPARING,token)
        self.pubs["kick"].publish(old)
        wait(lambda:len(self.router.rejected)==1)
        time.sleep(.03)
        self.assertFalse(self.received)

    def test_failure_routes_only_hcm(self):
        token=self.router.begin_transaction()
        self.router.acknowledge_phase(Phase.STOPPING,token)
        self.router.acknowledge_phase(Phase.PREPARING,token)
        self.router.fail_transaction("fall")
        self.assertFalse(self.router.acknowledge_phase(Phase.KICKING,token))
        self.bombard()
        self.assertEqual(self.router.gate.owner,"hcm")

    def test_malformed_targets_rejected(self):
        msg=self.message()
        msg.positions=[float("nan")]*len(msg.joint_names)
        self.pubs["hold"].publish(msg)
        wait(lambda:len(self.router.rejected)==1)
        self.assertFalse(self.received)


if __name__=="__main__":
    unittest.main()
