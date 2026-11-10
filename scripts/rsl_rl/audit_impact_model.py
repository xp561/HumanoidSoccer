"""Read live Isaac masses/materials/offsets from the frozen evaluation configuration."""
import sys,argparse
sys.path.insert(0,'/home/tl/Downloads/HumanoidSoccer/scripts/rsl_rl')
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a);simulation_app=app.app
import json,pickle,hashlib,traceback
from pathlib import Path
import numpy as np
import gymnasium as gym
import soccer.tasks
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdPhysics,UsdShade
import omni.usd
R=Path('/home/tl/Downloads/HumanoidSoccer');O=R/'outputs/r1_impact_20261025';BASE=R/'logs/rsl_rl/r1_flat/2026-10-16_21-27-44_r1_init_curriculum_20261016'
def plain(x):
 if hasattr(x,'detach'):return x.detach().cpu().numpy().tolist()
 return str(x)
def viewspec(view):
 out={}
 for name in ['get_masses','get_inertias','get_material_properties','get_contact_offsets','get_rest_offsets']:
  try:out[name]=plain(getattr(view,name)())
  except Exception as e:out[name]={'unavailable':repr(e)}
 for name in ['max_shapes','count','prim_paths']:
  if hasattr(view,name):out[name]=getattr(view,name)
 return out
try:
 with (BASE/'params/env.pkl').open('rb') as f:cfg=pickle.load(f)
 cfg.seed=1024000;cfg.scene.num_envs=1;cfg.commands.motion.motion_files=[str(x) for x in sorted((R/'motions/r1-corrected').glob('*.npz'))]
 cfg.commands.motion.debug_vis=False;cfg.observations.policy.enable_corruption=False
 for name in ['physics_material','add_joint_default_pos','base_com','push_robot']:setattr(cfg.events,name,None)
 env=gym.make('Tracking-Flat-R1-SoccerMoving-RNN-v0',cfg=cfg);raw=env.unwrapped;raw.reset(seed=1024000)
 stage=omni.usd.get_context().get_stage();simview=SimulationManager.get_physics_sim_view()
 out={'state':'completed','config_pickle':str(BASE/'params/env.pkl'),'ball_asset':str(cfg.scene.soccer_ball.spawn.usd_path),'simulation_default_material':cfg.sim.physics_material.to_dict(),'terrain_material':cfg.scene.terrain.physics_material.to_dict(),'ball':viewspec(raw.scene['soccer_ball'].root_physx_view),'feet':{},'usd_colliders':[]}
 for foot in ['left_ankle_roll_link','right_ankle_roll_link']:
  out['feet'][foot]=viewspec(simview.create_rigid_body_view('/World/envs/env_0/Robot/'+foot))
 for prim in stage.Traverse():
  path=str(prim.GetPath())
  relevant=path.startswith('/World/ground') or path.startswith('/World/envs/env_0/SoccerBall') or any(path.startswith('/World/envs/env_0/Robot/'+foot) for foot in out['feet'])
  if not relevant or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
  material,rel=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
  attrs={at.GetName():at.Get() for at in prim.GetAttributes() if any(k in at.GetName().lower() for k in ['offset','collision','approximation','radius'])}
  item={'path':path,'type':prim.GetTypeName(),'attributes':attrs,'bound_material_path':str(material.GetPath()) if material else None}
  if material:
   item['material_attributes']={at.GetName():at.Get() for at in material.GetPrim().GetAttributes() if any(k in at.GetName().lower() for k in ['friction','restitution','combine'])}
  out['usd_colliders'].append(item)
 out['sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),BASE/'params/env.pkl',Path(cfg.scene.soccer_ball.spawn.usd_path)] if f.exists()}
 (O/'live_impact_audit.json').write_text(json.dumps(out,indent=2,default=plain));print('LIVE_IMPACT_AUDIT_COMPLETED',flush=True)
 env.close()
except BaseException as e:
 (O/'live_impact_audit_failed.json').write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc()},indent=2));raise
finally:simulation_app.close()
