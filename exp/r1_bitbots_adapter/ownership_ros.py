"""Isolated command multiplexer; no real HCM or walking controller is started."""
import json
import os
from threading import RLock
from pathlib import Path
import numpy as np
from rclpy.node import Node
from bitbots_msgs.msg import JointCommand
from std_msgs.msg import String
from .handoff import Handoff


class OwnershipRouter(Node):
    def __init__(self, *, walker_available=False, **kwargs):
        super().__init__("r1_motion_ownership", namespace="/r1_adapter_test", **kwargs)
        if self.get_namespace() != "/r1_adapter_test" or not os.environ.get("ROS_DOMAIN_ID"):
            raise RuntimeError("The ownership prototype requires an isolated namespace and domain")
        meta=json.loads((Path.cwd()/"logs/r1_logic_fix/r1_mujoco_formal_20261017/artifacts/metadata.json").read_text())
        self.names=tuple(meta["action_joint_names"])
        self.gate=Handoff(walker_available=walker_available)
        self.lock=RLock()
        self.forwarded=[]
        self.rejected=[]
        self.output=self.create_publisher(JointCommand,"leased_motor_goals",100)
        self.status=self.create_publisher(String,"motion_ownership",10)
        topics={"walk":"walking_motor_goals","hold":"hold_motor_goals","prepare":"prepare_motor_goals",
                "kick":"kick_motor_goals","recover":"recovery_motor_goals","hcm":"hcm_motor_goals"}
        for source,topic in topics.items():
            self.create_subscription(JointCommand,topic,lambda msg,source=source:self._route(source,msg),100)
        self.create_timer(.02,self._tick)

    def now_ns(self):
        return self.get_clock().now().nanoseconds

    def begin_transaction(self):
        with self.lock:
            return self.gate.begin(self.now_ns())

    def acknowledge_phase(self, phase, generation, **kwargs):
        with self.lock:
            return self.gate.acknowledge(phase,generation,self.now_ns(),**kwargs)

    def fail_transaction(self, reason):
        with self.lock:
            self.gate.fail(reason,self.now_ns())

    def _tick(self):
        with self.lock:
            try:
                self.gate.tick(self.now_ns())
            except ValueError:
                pass
        self.status.publish(String(data=json.dumps(dict(
            phase=self.gate.phase.value,owner=self.gate.owner,generation=self.gate.generation,
            reason=self.gate.reason))))

    def _route(self,source,msg):
        with self.lock:
            self._route_locked(source,msg)

    def _route_locked(self,source,msg):
        ns=msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
        with self.lock:
            try:
                allowed=self.gate.allows(source,ns,self.now_ns())
            except ValueError:
                allowed=False
        valid=len(msg.joint_names)==len(self.names) and set(msg.joint_names)==set(self.names)
        for name in ("positions","kp","kd","max_torques"):
            values=np.asarray(getattr(msg,name),float)
            valid=valid and values.shape==(len(self.names),) and np.isfinite(values).all()
        if allowed and valid:
            self.forwarded.append(dict(source=source,stamp_ns=ns,phase=self.gate.phase.value))
            self.output.publish(msg)
        else:
            self.rejected.append(dict(source=source,stamp_ns=ns,phase=self.gate.phase.value))
