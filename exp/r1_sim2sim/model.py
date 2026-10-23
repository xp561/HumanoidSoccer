"""Build a separate R1 MJCF from the same URDF used for Isaac training."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .interface import inverse_rotate, vector


def build_model(repo, output, metadata, *, ball_mass, sliding_friction):
    repo, output = Path(repo),Path(output)
    urdf = repo/"source/whole_body_tracking/soccer/assets/unitree_description/urdf/r1/main.urdf"
    tree = ET.parse(urdf)
    root = tree.getroot()
    compiler = root.find("mujoco/compiler")
    compiler.set("meshdir",str(urdf.parent))
    compiler.set("fusestatic","true")
    ET.SubElement(root,"link",name="world")
    joint = ET.SubElement(root,"joint",name="root_joint",type="floating")
    ET.SubElement(joint,"parent",link="world")
    ET.SubElement(joint,"child",link="pelvis_link")
    generated_urdf = output/"floating_r1.urdf"
    tree.write(generated_urdf)
    imported = mujoco.MjModel.from_xml_path(str(generated_urdf))
    imported_xml = output/"r1_imported.xml"
    mujoco.mj_saveLastXML(str(imported_xml),imported)
    tree = ET.parse(imported_xml)
    root = tree.getroot()
    option = root.find("option")
    if option is None: option = ET.SubElement(root,"option")
    option.set("timestep",str(metadata["physics_dt"]))
    option.set("gravity","0 0 -9.81")
    option.set("integrator","implicitfast")
    option.set("iterations","100")
    joints = {j.get("name"):j for j in root.findall(".//joint") if j.get("name")}
    for i,name in enumerate(metadata["joint_names"]):
        joints[name].set("armature",str(metadata["armature"][i]))
        joints[name].set("damping","0")
        joints[name].set("frictionloss","0")
    actuator = ET.SubElement(root,"actuator")
    for name in metadata["action_joint_names"]:
        i = metadata["joint_names"].index(name)
        effort = metadata["effort_limits"][i]
        ET.SubElement(actuator,"position",name=name,joint=name,kp=str(metadata["stiffness"][i]),
                      kv=str(metadata["damping"][i]),forcelimited="true",forcerange=f"{-effort} {effort}",ctrllimited="false")
    world = root.find("worldbody")
    ET.SubElement(world,"geom",name="ground",type="plane",size="10 10 .1",
                  friction="1 .005 .0001",rgba=".35 .4 .35 1")
    ball = ET.SubElement(world,"body",name="soccer_ball",pos="0 -1.5 .11")
    ET.SubElement(ball,"freejoint",name="soccer_ball_joint")
    ET.SubElement(ball,"geom",name="soccer_ball_geom",type="sphere",size=".11",mass=str(ball_mass),
                  friction=f"{sliding_friction} .005 .0001",rgba=".2 .9 .2 1")
    path = output/"r1_soccer.xml"
    tree.write(path)
    return path


class Robot:
    def __init__(self,path,metadata):
        self.meta = metadata
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.joint_ids = np.array([self.model.joint(n).id for n in metadata["joint_names"]])
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.dof_ids = self.model.jnt_dofadr[self.joint_ids]
        self.body_ids = np.array([self.model.body(n).id for n in metadata["body_names"]])
        self.action_ids = np.array([self.model.actuator(n).id for n in metadata["action_joint_names"]])
        self.root_id = self.model.body("pelvis_link").id
        self.root_q = int(self.model.joint("root_joint").qposadr[0])
        self.root_v = int(self.model.joint("root_joint").dofadr[0])
        self.ball_q = int(self.model.joint("soccer_ball_joint").qposadr[0])
        self.ball_v = int(self.model.joint("soccer_ball_joint").dofadr[0])
        self.foot_bodies = {self.model.body(n).id for n in ["left_ankle_roll_link","right_ankle_roll_link"]}

    def set_state(self, *, root_pos,root_quat,joint_pos,joint_vel,root_lin_vel,root_ang_vel,
                  ball_pos=(0,-1.5,.11),ball_vel=(0,0,0)):
        mujoco.mj_resetData(self.model,self.data)
        q,v = self.root_q,self.root_v
        self.data.qpos[q:q+3] = root_pos
        self.data.qpos[q+3:q+7] = root_quat
        self.data.qpos[self.qpos_ids] = vector(joint_pos,len(self.qpos_ids),"joints")
        self.data.qvel[self.dof_ids] = vector(joint_vel,len(self.dof_ids),"velocities")
        # Isaac's legacy root state stores COM linear velocity, whereas the
        # free-joint translation is the link-origin position in MuJoCo.
        com_offset=np.empty(3)
        mujoco.mju_rotVecQuat(com_offset,self.model.body_ipos[self.root_id],self.data.qpos[q+3:q+7])
        self.data.qvel[v:v+3] = np.asarray(root_lin_vel)-np.cross(root_ang_vel,com_offset)
        self.data.qvel[v+3:v+6] = inverse_rotate(root_quat,root_ang_vel)
        self.data.qpos[self.ball_q:self.ball_q+3] = ball_pos
        self.data.qpos[self.ball_q+3:self.ball_q+7] = [1,0,0,0]
        self.data.qvel[self.ball_v:self.ball_v+3] = ball_vel
        mujoco.mj_forward(self.model,self.data)

    def action(self,actions):
        a = vector(actions,len(self.action_ids),"actions")
        self.data.ctrl[self.action_ids] = np.asarray(self.meta["action_offset"])+np.asarray(self.meta["action_scale"])*a

    def state(self):
        # mj_step integrates qpos after updating the Cartesian body caches.
        # The floating root's generalized pose is the current world pose.
        q = self.data.qpos[self.root_q+3:self.root_q+7].copy()
        world_angular = np.empty(3)
        mujoco.mju_rotVecQuat(world_angular,self.data.qvel[self.root_v+3:self.root_v+6],q)
        return dict(root_pos=self.data.qpos[self.root_q:self.root_q+3].copy(),root_quat=q,
                    root_ang_vel=world_angular,joint_pos=self.data.qpos[self.qpos_ids].copy(),
                    joint_vel=self.data.qvel[self.dof_ids].copy(),ball_pos=self.data.qpos[self.ball_q:self.ball_q+3].copy())

    def ball_foot_contact(self):
        ball = self.model.geom("soccer_ball_geom").id
        return any((c.geom1==ball and self.model.geom_bodyid[c.geom2] in self.foot_bodies)
                   or (c.geom2==ball and self.model.geom_bodyid[c.geom1] in self.foot_bodies)
                   for c in self.data.contact)
