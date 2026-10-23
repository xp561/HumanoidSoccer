"""Independent boundary checks for the captured evaluation semantics."""
import unittest
import importlib.util
import tempfile
from pathlib import Path
import numpy as np
from exp.r1_sim2sim.protocol import sample_initial, noisy_observation, termination_values


def runtime():
    return {"reference_body_names":["pelvis_link","foot"],
            "robot_body_names":["foot","pelvis_link"],
            "control_dt":.02,"episode_length_s":10.,
            "terminations":{
                "height":{"function":"bad_anchor_pos_z_only","params":{"threshold":.25}},
                "orientation":{"function":"bad_anchor_ori","params":{"threshold":.8}},
                "foot":{"function":"bad_motion_body_pos_z_only","params":{"threshold":.25,"body_names":["foot"]}},
                "timeout":{"function":"time_out","params":{}}}}


class ProtocolTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("mujoco") is not None,"Requires MuJoCo")
    def test_contact_probes_never_activate_visual_meshes(self):
        from exp.r1_sim2sim.contact_probe import scene
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/"model.xml"
            source.write_text('<mujoco><compiler/><asset/><worldbody><body name="left_ankle_roll_link"><geom type="sphere" size=".1" contype="0" conaffinity="0"/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
            root=scene(source,impact=True)
            self.assertEqual(len(root.findall("contact/pair")),1)
            self.assertEqual(len(root.findall(".//geom")),2)

    def test_disk_area_and_speed_distribution(self):
        rng=np.random.default_rng(42)
        samples=[sample_initial(rng,.5,(.1,.3)) for _ in range(10000)]
        xy=np.stack([s[0] for s in samples]); v=np.stack([s[1] for s in samples])
        radius2=np.sum(xy**2,axis=1); speed=np.linalg.norm(v,axis=1)
        self.assertLessEqual(radius2.max(),.25)
        self.assertAlmostEqual(radius2.mean(),.125,delta=.003)
        self.assertAlmostEqual(speed.mean(),.2,delta=.002)
        self.assertLess(np.linalg.norm(xy.mean(axis=0)),.008)

    def test_invalid_sampling(self):
        for radius,speeds in [(-1,(0,0)),(1,(-1,1)),(1,(2,1))]:
            with self.assertRaises(ValueError):sample_initial(np.random.default_rng(0),radius,speeds)

    def test_named_bodies_and_strict_thresholds(self):
        r=runtime(); ref=np.array([[0,0,1],[0,0,0.]])
        body=ref[::-1].copy(); body[0,2]=.25
        flags,_=termination_values(r,[0,0,.75],[1,0,0,0],body,ref,np.array([[1,0,0,0]]*2),499)
        self.assertFalse(any(flags.values()))
        body[0,2]+=.001
        flags,_=termination_values(r,[0,0,.749],[1,0,0,0],body,ref,np.array([[1,0,0,0]]*2),500)
        self.assertTrue(flags["height"] and flags["foot"] and flags["timeout"])

    def test_orientation_uses_projected_gravity_not_yaw(self):
        r=runtime(); ref=np.array([[0,0,1],[0,0,0.]])
        for q,expected in [([0,0,0,1],False),([0,1,0,0],True)]:
            flags,_=termination_values(r,[0,0,1],q,ref[::-1],ref,np.array([[1,0,0,0]]*2),1)
            self.assertEqual(flags["orientation"],expected)

    def test_noise_uses_named_slices_and_preserves_input(self):
        terms=["command","projected_gravity","motion_ref_ang_vel","base_ang_vel","joint_pos","joint_vel","actions","target_point_pos","target_destination_pos_local"]
        meta={"joint_names":["j"],"action_joint_names":["j"],"observation_terms":terms}
        r={"enable_corruption":True,"noise":{name:None for name in terms}}
        r["noise"]["projected_gravity"]={"type":"UniformNoiseCfg","operation":"add","min":-.05,"max":.05}
        x=np.zeros(20,dtype=np.float32)
        y=noisy_observation(x,meta,r,np.random.default_rng(0))
        self.assertTrue(np.any(y[2:5]!=0)); self.assertTrue(np.all(np.abs(y[2:5])<=.05))
        self.assertTrue(np.all(np.concatenate([y[:2],y[5:]])==0))
        self.assertTrue(np.all(x==0))

    def test_unsupported_termination_is_rejected(self):
        r=runtime(); r["terminations"]["height"]["function"]="unknown"
        with self.assertRaisesRegex(ValueError,"Unsupported"):
            termination_values(r,None,None,None,None,None,0)


if __name__ == "__main__": unittest.main()
