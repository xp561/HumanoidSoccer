"""Interface failure modes must be explicit rather than silently padded."""
import unittest
import importlib.util
import numpy as np
from exp.r1_sim2sim.interface import inverse_rotate,observation,vector


class InterfaceTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("mujoco") is not None,"Requires MuJoCo inference environment")
    def test_reset_preserves_center_of_mass_velocity(self):
        import mujoco
        from exp.r1_sim2sim.model import Robot
        r=object.__new__(Robot)
        r.model=mujoco.MjModel.from_xml_string('<mujoco><option gravity="0 0 0"/><worldbody><body name="root"><freejoint/><geom type="sphere" size=".1" mass="1" pos=".1 0 0"/></body><body><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
        r.data=mujoco.MjData(r.model);r.root_q=r.root_v=0;r.root_id=1;r.ball_q=7;r.ball_v=6
        r.qpos_ids=r.dof_ids=np.array([],dtype=int)
        r.set_state(root_pos=[0,0,1],root_quat=[1,0,0,0],joint_pos=[],joint_vel=[],
                    root_lin_vel=[0,0,0],root_ang_vel=[0,0,1])
        velocity=np.empty(6)
        mujoco.mj_objectVelocity(r.model,r.data,mujoco.mjtObj.mjOBJ_BODY,r.root_id,velocity,0)
        np.testing.assert_allclose(velocity[3:],[0,0,0],atol=1e-12)

    @unittest.skipUnless(importlib.util.find_spec("mujoco") is not None,"Requires MuJoCo inference environment")
    def test_root_observation_uses_integrated_pose(self):
        import mujoco
        from exp.r1_sim2sim.model import Robot
        r=object.__new__(Robot)
        r.model=mujoco.MjModel.from_xml_string('<mujoco><option gravity="0 0 0"/><worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
        r.data=mujoco.MjData(r.model)
        r.root_q=r.root_v=r.ball_q=0;r.root_id=1
        r.qpos_ids=r.dof_ids=np.array([],dtype=int)
        r.data.qvel[0]=1
        mujoco.mj_forward(r.model,r.data)
        mujoco.mj_step(r.model,r.data)
        self.assertGreater(r.data.qpos[0],r.data.xpos[1,0])
        np.testing.assert_allclose(r.state()["root_pos"],r.data.qpos[:3])

    def test_inverse_rotation_identity(self):
        np.testing.assert_allclose(inverse_rotate([1,0,0,0],[1,2,3]),[1,2,3])

    def test_inverse_rotation_quarter_turn(self):
        q=[np.sqrt(.5),0,0,np.sqrt(.5)]
        np.testing.assert_allclose(inverse_rotate(q,[1,0,0]),[0,-1,0],atol=1e-12)

    def test_quaternion_sign_and_normalization(self):
        q=np.array([.7,.2,.3,.4])
        np.testing.assert_allclose(inverse_rotate(q,[1,2,3]),inverse_rotate(-3*q,[1,2,3]))

    def test_invalid_rotation(self):
        for q in [[0,0,0,0],[np.nan,0,0,0],[1,0,0]]:
            with self.subTest(q=q),self.assertRaises(ValueError):inverse_rotate(q,[1,2,3])

    def test_no_silent_dimension_repair(self):
        for x in [[1,2],[1,2,3,4],[1,np.inf,3]]:
            with self.subTest(x=x),self.assertRaises(ValueError):vector(x,3,"input")

    def test_observation_order_is_validated(self):
        with self.assertRaisesRegex(ValueError,"ordering"):
            observation({"observation_terms":[]},ref_joint_pos=[],ref_joint_vel=[],ref_ang_vel=[],
                        root_pos=[],root_quat=[],root_ang_vel=[],joint_pos=[],joint_vel=[],
                        last_action=[],ball_pos=[],goal_pos=[])


if __name__ == "__main__":unittest.main()
