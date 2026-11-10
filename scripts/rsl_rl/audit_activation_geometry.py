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
from pxr import UsdPhysics,UsdShade,UsdGeom,Usd,Gf,PhysxSchema
import omni.usd
R=Path('/home/tl/Downloads/HumanoidSoccer');O=R/'outputs/r1_activation_geometry_20261026';BASE=R/'logs/rsl_rl/r1_flat/2026-10-16_21-27-44_r1_init_curriculum_20261016'
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
 out['foot_meshes']={};out['cooking_queries']=[];out['ground_contact_schema']=[]
 cache=UsdGeom.XformCache()
 for foot in ['left_ankle_roll_link','right_ankle_roll_link']:
  base=stage.GetPrimAtPath('/World/envs/env_0/Robot/'+foot);meshes=[]
  for prim in Usd.PrimRange(base,Usd.TraverseInstanceProxies()):
   path=str(prim.GetPath())
   if 'collision' not in path.lower() or not prim.IsA(UsdGeom.Mesh):continue
   points=UsdGeom.Mesh(prim).GetPointsAttr().Get();transform,_=cache.ComputeRelativeTransform(prim,base);vertices=[list(transform.Transform(Gf.Vec3d(float(v[0]),float(v[1]),float(v[2])))) for v in points]
   meshes.append({'path':path,'vertex_count':len(vertices),'vertices_in_foot_link':vertices,'mesh_bounds_in_foot_link':[np.min(vertices,axis=0).tolist(),np.max(vertices,axis=0).tolist()]})
  out['foot_meshes'][foot]=meshes
 for prim in stage.Traverse():
  path=str(prim.GetPath())
  if path.startswith('/World/ground') and prim.HasAPI(UsdPhysics.CollisionAPI):
   api=PhysxSchema.PhysxCollisionAPI(prim);out['ground_contact_schema'].append({'path':path,'contact_offset':api.GetContactOffsetAttr().Get(),'rest_offset':api.GetRestOffsetAttr().Get()})
 try:
  import omni.physx
  cooking=omni.physx.get_physx_cooking_interface();out['cooking_getters']=[n for n in dir(cooking) if 'convex' in n.lower()]
  candidates={v['path'] for meshes in out['foot_meshes'].values() for v in meshes}|{v['path'] for v in out['usd_colliders'] if 'ankle_roll_link' in v['path']}
  for path in sorted(candidates):
   item={'path':path}
   try:
    count=cooking.get_nb_convex_mesh_data(path);item['convex_count']=count;item['convexes']=[]
    for i in range(count):
     data=cooking.get_convex_mesh_data(path,i);record={'type':str(type(data))}
     if isinstance(data,dict):
      record['keys']=list(data);vertices=data.get('vertices')
     else:record['attributes']=[n for n in dir(data) if not n.startswith('_')];vertices=getattr(data,'vertices',None)
     if vertices is not None:record['vertices_raw']=[[float(v[0]),float(v[1]),float(v[2])] for v in vertices]
     item['convexes'].append(record)
   except Exception as e:item['unavailable']=repr(e)
   out['cooking_queries'].append(item)
 except Exception as e:out['cooking_unavailable']=repr(e)
 out['sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),BASE/'params/env.pkl',Path(cfg.scene.soccer_ball.spawn.usd_path)] if f.exists()}
 (O/'live_impact_audit.json').write_text(json.dumps(out,indent=2,default=plain));print('LIVE_GEOMETRY_AUDIT_COMPLETED',flush=True)
 env.close()
except BaseException as e:
 (O/'live_impact_audit_failed.json').write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc()},indent=2));raise
finally:simulation_app.close()
