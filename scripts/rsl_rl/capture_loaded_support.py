"""Exact-state Isaac capture with explicit shared actor observation semantics."""
import argparse,sys
from pathlib import Path
R=Path('/home/tl/Downloads/HumanoidSoccer');sys.path.insert(0,str(R));sys.path.insert(0,str(R/'scripts/rsl_rl'))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--num_envs',type=int,default=16);p.add_argument('--smoke-only',action='store_true');AppLauncher.add_app_launcher_args(p);a=p.parse_args();launcher=AppLauncher(a);simulation_app=launcher.app
import json,pickle,hashlib
import numpy as np
import torch,gymnasium as gym
import soccer.tasks
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdPhysics
import omni.usd,re
from exp.r1_sim2sim.interface import observation
from exp.r1_sim2sim.protocol import noisy_observation
from exp.r1_transfer.ball_observation_guard import bounded_ball_observation
S=R/'outputs/r1_direction_pair_20261025';O=Path(a.output);M=json.loads((S/'manifest.json').read_text());B=R/'logs/rsl_rl/r1_flat/2026-10-16_21-27-44_r1_init_curriculum_20261016'
def arr(x):return x.detach().cpu().numpy().copy()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def main():
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 with (B/'params/env.pkl').open('rb') as f:cfg=pickle.load(f)
 cfg.seed=20261025;cfg.scene.num_envs=a.num_envs;cfg.scene.env_spacing=4.
 cfg.commands.motion.motion_files=[str(p) for p in sorted((R/'motions/r1-corrected').glob('*.npz'))]
 c=cfg.commands.motion;c.debug_vis=False;c.eval_single_attempt=True;c.eval_robot_init_radius=0.;c.eval_match_motion_to_ball=True;c.eval_fixed_ball_pos=(0.,-1.,.11);c.eval_fixed_ball_lin_vel=(0.,0.,0.);c.eval_fixed_target_destination=(0.,-5.,.11);c.eval_ball_speed_range=(0.,0.)
 c.pose_range={k:(0.,0.) for k in ['x','y','z','roll','pitch','yaw']};c.velocity_range=dict(c.pose_range);c.joint_position_range=(0.,0.)
 for k in ['physics_material','add_joint_default_pos','base_com','push_robot']:setattr(cfg.events,k,None)
 cfg.observations.policy.enable_corruption=False
 for k in list(vars(cfg.terminations)):
  if not k.startswith('_'):setattr(cfg.terminations,k,None)
 env=gym.make('Tracking-Flat-R1-SoccerMoving-RNN-v0',cfg=cfg);raw=env.unwrapped;robot=raw.scene['robot'];ball=raw.scene['soccer_ball'];cmd=raw.command_manager.get_term('motion');sensor=raw.scene['soccer_ball_contact']
 meta=json.loads((R/'outputs/r1_core_contact_20261025/artifacts/metadata.json').read_text());runtime=json.loads((R/'outputs/r1_core_contact_20261025/runtime_protocol.json').read_text());runtime['enable_corruption']=True
 assert robot.joint_names==meta['joint_names'] and robot.body_names==meta['body_names'];assert raw.step_dt==.02 and raw.physics_dt==.005
 np.testing.assert_allclose(arr(robot.data.default_joint_pos)[0],meta['default_joint_pos'],rtol=0,atol=1e-6)
 feet=[robot.body_names.index(n) for n in ['left_ankle_roll_link','right_ankle_roll_link']];action_ids=meta['action_joint_ids']
 stage=omni.usd.get_context().get_stage()
 ground_paths=[str(prim.GetPath()) for prim in stage.Traverse() if str(prim.GetPath()).startswith('/World/ground/') and prim.HasAPI(UsdPhysics.CollisionAPI)]
 assert ground_paths, 'No ground collision prim found'
 simview=SimulationManager.get_physics_sim_view();support_views=[];support_orders=[]
 for footname in ['left_ankle_roll_link','right_ankle_roll_link']:
  view=simview.create_rigid_contact_view('/World/envs/env_*/Robot/'+footname,filter_patterns=ground_paths,max_contact_data_count=raw.num_envs*64)
  assert view.sensor_count==raw.num_envs and view.filter_count>0
  env_ids=[int(re.search(r'/env_(\d+)/',path).group(1)) for path in view.sensor_paths]
  assert sorted(env_ids)==list(range(raw.num_envs))
  support_views.append(view);support_orders.append(np.argsort(env_ids))

 audit={'joint_names':meta['joint_names'],'body_names':meta['body_names'],'data_fields':{},'physx_fields':{},'physx_getters':[n for n in dir(robot.root_physx_view) if n.startswith('get_') and any(k in n for k in ['dof','mass','inertia','com'])],'torque_note':'Implicit applied_torque is an estimated PD quantity, not measured motor torque.'}
 for name in ['joint_stiffness','joint_damping','joint_armature','joint_friction_coeff','joint_effort_limits','joint_vel_limits','default_mass','default_inertia']:
  if hasattr(robot.data,name):audit['data_fields'][name]=arr(getattr(robot.data,name))[0].tolist()
 for name in ['get_dof_stiffnesses','get_dof_dampings','get_dof_armatures','get_dof_max_forces','get_dof_max_velocities','get_dof_friction_coefficients','get_masses','get_inertias','get_coms']:
  if hasattr(robot.root_physx_view,name):audit['physx_fields'][name]=arr(getattr(robot.root_physx_view,name)())[0].tolist()
 collision=[]
 for prim in stage.Traverse():
  path=str(prim.GetPath())
  if path.startswith('/World/envs/env_0/Robot/') and prim.HasAPI(UsdPhysics.CollisionAPI) and any(n in path for n in ['left_ankle_roll_link','right_ankle_roll_link']):
   collision.append({'path':path,'type':str(prim.GetTypeName()),'attributes':{n:str(prim.GetAttribute(n).Get()) for n in ['physxCollision:contactOffset','physxCollision:restOffset','physics:approximation','physics:collisionEnabled'] if prim.GetAttribute(n)}})
 audit['foot_collision_prims']=collision;write(O/'native_parameters.json',audit)

 def support_values():
  normals=[];tangents=[];centers=[];slips=[];counts_all=[];sep_all=[];minz_all=[]
  com=arr(robot.data.body_com_pos_w[:,feet]);vel=arr(robot.data.body_com_lin_vel_w[:,feet]);ang=arr(robot.data.body_com_ang_vel_w[:,feet])
  for fi,(view,order) in enumerate(zip(support_views,support_orders)):
   nf=arr(view.get_contact_force_matrix(raw.physics_dt)).sum(axis=1)[order]
   forces,points,normal,separation,counts,starts=view.get_contact_data(raw.physics_dt)
   points=arr(points);separation=arr(separation);counts=arr(counts);starts=arr(starts)
   friction,_,fc,fs=view.get_friction_data(raw.physics_dt)
   friction=arr(friction);fc=arr(fc);fs=arr(fs)
   assert counts.sum()<view.max_contact_data_count and fc.sum()<view.max_contact_data_count, 'Contact buffer may be truncated'
   tf=np.zeros((raw.num_envs,3));center=np.full_like(tf,np.nan);slip=np.full(raw.num_envs,np.nan);ct=np.zeros(raw.num_envs,dtype=np.int32)
   for e,source in enumerate(order):
    locations=[];depths=[]
    for k in range(view.filter_count):
     start=int(starts[source,k]);n=int(counts[source,k]);ct[e]+=n
     if n:locations.append(points[start:start+n]);depths.append(separation[start:start+n])
     start=int(fs[source,k]);n=int(fc[source,k])
     if n:tf[e]+=friction[start:start+n].sum(axis=0)
    if locations:
     pos=np.concatenate(locations);center[e]=pos.mean(axis=0)-arr(raw.scene.env_origins)[e]
     pointvel=vel[e,fi]+np.cross(ang[e,fi],pos-com[e,fi])
     slip[e]=np.linalg.norm(pointvel[:,:2],axis=1).max()
   sep_all.append(np.array([np.nan if int(counts[source].sum())==0 else min(float(separation[int(starts[source,k]):int(starts[source,k])+int(counts[source,k])].min()) for k in range(view.filter_count) if counts[source,k]>0) for source in order]));minz_all.append(center[:,2].copy());normals.append(nf);tangents.append(tf);centers.append(center);slips.append(slip);counts_all.append(ct)
  return {'ground_normal_force':np.stack(normals,axis=1),'ground_tangent_force':np.stack(tangents,axis=1),'ground_contact_center':np.stack(centers,axis=1),'ground_point_slip_max':np.stack(slips,axis=1),'ground_contact_count':np.stack(counts_all,axis=1),'ground_min_separation':np.stack(sep_all,axis=1),'ground_center_z':np.stack(minz_all,axis=1)}
 capture={'active':False};original=raw.scene.update
 def update(dt):
  original(dt)
  if capture['active']:
   org=raw.scene.env_origins
   for k,v in {'ball_pos':ball.data.root_pos_w-org,'ball_vel':ball.data.root_lin_vel_w,'foot_pos':robot.data.body_pos_w[:,feet]-org[:,None,:],'contact_force':sensor.data.force_matrix_w}.items():capture.setdefault(k,[]).append(arr(v))
   for k,v in support_values().items():capture.setdefault(k,[]).append(v.copy())
   for k,v in {'joint_pos_physics':robot.data.joint_pos,'joint_vel_physics':robot.data.joint_vel,'joint_target':robot.data.joint_pos_target,'estimated_pd_torque':robot.data.applied_torque,'foot_quat':robot.data.body_quat_w[:,feet],'body_com_vel':robot.data.body_com_lin_vel_w,'root_com_velocity':robot.data.root_lin_vel_w}.items():capture.setdefault(k,[]).append(arr(v))
 raw.scene.update=update
 gates=[];items=M['items'][:2] if a.smoke_only else M['items']
 try:
  for start in range(0,len(items),a.num_envs):
   batch=items[start:start+a.num_envs];padded=batch+[batch[-1]]*(a.num_envs-len(batch));packs=[dict(np.load(S/'inputs'/f"case_{x['case']['case']:05d}.npz")) for x in padded]
   env.reset(seed=20261025+start);raw.action_manager.reset();origins=arr(raw.scene.env_origins)
   root=np.stack([np.r_[d['initial_root_pos'],d['initial_root_quat'],d['initial_root_lin_vel'],d['initial_root_ang_vel']] for d in packs]).astype(np.float32);root[:,:3]+=origins
   bstate=np.stack([np.r_[d['initial_ball_pos'],[1.,0.,0.,0.],d['initial_ball_vel'],[0.,0.,0.]] for d in packs]).astype(np.float32);bstate[:,:3]+=origins
   tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=raw.device)
   robot.write_joint_state_to_sim(tensor(np.stack([d['initial_joint_pos'] for d in packs])),tensor(np.stack([d['initial_joint_vel'] for d in packs])))
   robot.write_root_state_to_sim(tensor(root));ball.write_root_state_to_sim(tensor(bstate));raw.scene.reset();raw.sim.forward();raw.scene.update(0.)
   cmd.motion_idx[:]=torch.tensor([cmd.motion.motion_name.index(x['expected']['motion']) for x in padded],device=raw.device);cmd.motion_length[:]=cmd.motion.file_lengths[cmd.motion_idx];cmd.time_steps[:]=0;cmd.target_destination_pos[:]=tensor(np.tile([0.,-5.,.11],(a.num_envs,1)));cmd._update_motion_alignment()
   sources=[dict(np.load(S/'isaac'/f"case_{x['case']['case']:05d}.npz")) for x in padded]
   records={};capture.clear();capture['active']=False;errors={'root_position_m':0.,'joint_position_rad':0.,'ball_position_m':0.,'action_target_rad':0.}
   for step in range(20):
    state={'root_pos':arr(robot.data.root_pos_w)-origins,'root_quat':arr(robot.data.root_quat_w),'root_ang_vel':arr(robot.data.root_ang_vel_w),'joint_pos':arr(robot.data.joint_pos),'joint_vel':arr(robot.data.joint_vel),'ball_pos':arr(ball.data.root_pos_w)-origins}
    for e in range(len(batch)):
     for key,label in [('root_pos','root_position_m'),('joint_pos','joint_position_rad'),('ball_pos','ball_position_m')]:errors[label]=max(errors[label],float(abs(state[key][e]-sources[e][key][step]).max()))
    assert max(errors.values())<2e-5,'Read-only fixed-action recapture differs: '+str(errors)
    actions=np.stack([d['actions'][step] for d in sources])
    for k,v in {**state,'actions':actions}.items():records.setdefault(k,[]).append(v.copy())
    capture['active']=True
    _,_,terminated,truncated,_=env.step(tensor(actions))
    capture['active']=False
    assert not (terminated.any() or truncated.any())
    expected=np.asarray(meta['action_offset'])+np.asarray(meta['action_scale'])*actions
    errors['action_target_rad']=max(errors['action_target_rad'],float(abs(arr(robot.data.joint_pos_target)[:,action_ids]-expected).max()));assert errors['action_target_rad']<2e-5
    for k,v in {'post_root_pos':robot.data.root_pos_w-raw.scene.env_origins,'post_root_quat':robot.data.root_quat_w,'post_body_pos':robot.data.body_pos_w-raw.scene.env_origins[:,None,:]}.items():records.setdefault(k,[]).append(arr(v))
   gates.append({'batch_start':start,'cases':len(batch),'max_errors':errors});write(O/'isaac/reproduction_gates.json',{'passed':True,'batches':gates})
   data={k:np.stack(v) for k,v in records.items()};physics={k:np.stack(v) for k,v in capture.items() if k!='active'};assert len(physics['ball_pos'])==80
   for e,x in enumerate(batch):
    idx=x['case']['case'];dest=O/'isaac'/f'case_{idx:05d}.npz';dest.parent.mkdir(exist_ok=True)
    np.savez_compressed(dest,**{k:v[:,e] for k,v in data.items()},**{'physics_'+k:v[:,e] for k,v in physics.items()})
   write(O/'isaac/status.json',{'state':'running','completed_cases':start+len(batch),'total_cases':len(items),'reproduction_max_errors':errors});print('direction pairs captured',start+len(batch),'/',len(items),flush=True)
  write(O/'isaac/status.json',{'state':'completed','completed_cases':len(items),'total_cases':len(items)})
 finally:env.close()
try:main()
except BaseException as e:
 import traceback
 write(O/'isaac/failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
finally:simulation_app.close()
