"""Read-only support-contact instrumentation; preserves shared actor and rollout protocol."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,'/home/tl/Downloads/HumanoidSoccer/scripts/rsl_rl')
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--actor-socket',required=True);p.add_argument('--strict-fp32',action='store_true');p.add_argument('--batch-index',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--num_envs',type=int,default=32);p.add_argument('--steps',type=int,default=400)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a);simulation_app=app.app
import gymnasium as gym
import numpy as np
import torch,json,pickle,hashlib
import soccer.tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdPhysics
import omni.usd,re
from rsl_rl.runners import OnPolicyRunner
R=Path('/home/tl/Downloads/HumanoidSoccer');O=Path(a.output);O.mkdir(parents=True,exist_ok=True)
BASE=R/'logs/rsl_rl/r1_flat/2026-10-16_21-27-44_r1_init_curriculum_20261016';CHECKPOINT=BASE/'model_107996.pt'
TASK='Tracking-Flat-R1-SoccerMoving-RNN-v0'
if a.strict_fp32:
 torch.backends.cudnn.allow_tf32=False
 torch.backends.cuda.matmul.allow_tf32=False

def arr(x):return x.detach().cpu().numpy().copy()
def main():
 with (BASE/'params/env.pkl').open('rb') as f:cfg=pickle.load(f)
 with (BASE/'params/agent.pkl').open('rb') as f:agent=pickle.load(f)
 cfg.seed=1024000;cfg.scene.num_envs=a.num_envs;cfg.commands.motion.motion_files=[str(x) for x in sorted((R/'motions/r1-corrected').glob('*.npz'))]
 cmdcfg=cfg.commands.motion;cmdcfg.debug_vis=False;cmdcfg.eval_single_attempt=True;cmdcfg.eval_robot_init_radius=.5;cmdcfg.eval_match_motion_to_ball=True
 cmdcfg.eval_fixed_ball_pos=(.25,-.5,.11);cmdcfg.eval_fixed_ball_lin_vel=(0.,0.,0.);cmdcfg.eval_fixed_target_destination=(0.,-5.,.11);cmdcfg.eval_ball_speed_range=(0.,0.)
 cmdcfg.pose_range={k:(0.,0.) for k in ['x','y','z','roll','pitch','yaw']};cmdcfg.velocity_range=dict(cmdcfg.pose_range);cmdcfg.joint_position_range=(0.,0.)
 cfg.events.physics_material=None;cfg.events.add_joint_default_pos=None;cfg.events.base_com=None;cfg.events.push_robot=None
 cfg.observations.policy.enable_corruption=False
 # Prevent reset/early exit from changing the task: tracking failures are rescored separately.
 for name,value in list(vars(cfg.terminations).items()):
  if not name.startswith('_'):setattr(cfg.terminations,name,None)
 env=gym.make(TASK,cfg=cfg);raw=env.unwrapped;wrap=RslRlVecEnvWrapper(env)
 runner=OnPolicyRunner(wrap,agent.to_dict(),log_dir=None,device=raw.device);runner.load(str(CHECKPOINT));policy=runner.get_inference_policy(device=raw.device)
 robot=raw.scene['robot'];ball=raw.scene['soccer_ball'];cmd=raw.command_manager.get_term('motion');sensor=raw.scene['soccer_ball_contact']
 cmdcfg=cmd.cfg
 feet=[robot.body_names.index(x) for x in ['left_ankle_roll_link','right_ankle_roll_link']]
 metadata=json.loads((R/'logs/r1_logic_fix/r1_mujoco_formal_20261017/artifacts/metadata.json').read_text())
 assert robot.joint_names==metadata['joint_names'] and robot.body_names==metadata['body_names']
 assert raw.step_dt==.02 and raw.physics_dt==.005
 from multiprocessing.connection import Client
 connection=Client(a.actor_socket,family='AF_UNIX',authkey=b'r1-paired-diagnostic')
 # Direct tensor views expose tangential forces as well as normal contacts.
 # Exact collision prim filters exclude ball and robot self-contact.
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
 (O/'support_schema.json').write_text(json.dumps({'ground_collision_paths':ground_paths,'views':[{'sensor_paths':list(v.sensor_paths),'filter_paths':list(v.filter_paths),'capacity':v.max_contact_data_count} for v in support_views],
  'normal_force':'filtered PhysX contact normal force in world frame, N',
  'tangent_force':'sum of get_friction_data forces in world frame, N',
  'point_slip':'maximum horizontal rigid-body velocity evaluated at reported ground contact points; NaN when no contact',
  'estimated_pd_torque':'Isaac implicit applied_torque is an approximate clipped PD calculation, NOT measured motor torque',
  'physics_time':'post-physics states at (index+1)*0.005 seconds'},indent=2))
 capture={'active':False,'physics':{}}
 def support_values():
  normals=[];tangents=[];centers=[];slips=[];counts_all=[]
  com=arr(robot.data.body_com_pos_w[:,feet]);vel=arr(robot.data.body_com_lin_vel_w[:,feet]);ang=arr(robot.data.body_com_ang_vel_w[:,feet])
  for fi,(view,order) in enumerate(zip(support_views,support_orders)):
   nf=arr(view.get_contact_force_matrix(raw.physics_dt)).sum(axis=1)[order]
   forces,points,normal,separation,counts,starts=view.get_contact_data(raw.physics_dt)
   points=arr(points);counts=arr(counts);starts=arr(starts)
   friction,_,fc,fs=view.get_friction_data(raw.physics_dt)
   friction=arr(friction);fc=arr(fc);fs=arr(fs)
   assert counts.sum()<view.max_contact_data_count and fc.sum()<view.max_contact_data_count, 'Contact buffer may be truncated'
   tf=np.zeros((raw.num_envs,3));center=np.full_like(tf,np.nan);slip=np.full(raw.num_envs,np.nan);ct=np.zeros(raw.num_envs,dtype=np.int32)
   for e,source in enumerate(order):
    locations=[]
    for k in range(view.filter_count):
     start=int(starts[source,k]);n=int(counts[source,k]);ct[e]+=n
     if n:locations.append(points[start:start+n])
     start=int(fs[source,k]);n=int(fc[source,k])
     if n:tf[e]+=friction[start:start+n].sum(axis=0)
    if locations:
     pos=np.concatenate(locations);center[e]=pos.mean(axis=0)-arr(raw.scene.env_origins)[e]
     pointvel=vel[e,fi]+np.cross(ang[e,fi],pos-com[e,fi])
     slip[e]=np.linalg.norm(pointvel[:,:2],axis=1).max()
   normals.append(nf);tangents.append(tf);centers.append(center);slips.append(slip);counts_all.append(ct)
  return {'ground_normal_force':np.stack(normals,axis=1),'ground_tangent_force':np.stack(tangents,axis=1),'ground_contact_center':np.stack(centers,axis=1),'ground_point_slip_max':np.stack(slips,axis=1),'ground_contact_count':np.stack(counts_all,axis=1)}
 def add(records,key,x):records.setdefault(key,[]).append(arr(x))
 original_update=raw.scene.update
 def update(dt):
  original_update(dt)
  if not capture['active']:return
  records=capture['physics'];org=raw.scene.env_origins
  values={'ball_pos':ball.data.root_pos_w-org,'ball_vel':ball.data.root_lin_vel_w,'ball_ang_vel':ball.data.root_ang_vel_w,
   'foot_pos':robot.data.body_pos_w[:,feet]-org[:,None,:],'foot_quat':robot.data.body_quat_w[:,feet],
   'foot_lin_vel':robot.data.body_lin_vel_w[:,feet],'foot_ang_vel':robot.data.body_ang_vel_w[:,feet],
   'contact_force':sensor.data.force_matrix_w}
  for k,v in values.items():add(records,k,v)
  for k,v in support_values().items():records.setdefault(k,[]).append(v.copy())
  for k,v in {'joint_pos_physics':robot.data.joint_pos,'joint_vel_physics':robot.data.joint_vel,'estimated_pd_torque':robot.data.applied_torque,'joint_target':robot.data.joint_pos_target,'foot_com_pos':robot.data.body_com_pos_w[:,feet]-org[:,None,:],'foot_com_vel':robot.data.body_com_lin_vel_w[:,feet]}.items():add(records,k,v)
 raw.scene.update=update
 batches=[]
 try:
  for pi,(forward,lateral) in enumerate([(.5,.25),(1.,-1.),(1.,1.),(1.5,0.)]):
   for ci,condition in enumerate(['static','rolling']):
    if pi*2+ci != a.batch_index:continue
    tag=f'p{pi}_{condition}';seed=1024000+pi*100+ci*10
    cmdcfg.eval_fixed_ball_pos=(lateral,-forward,.11);cmdcfg.eval_ball_speed_range=(0.,0.) if ci==0 else (.1,.3)
    raw.reset(seed=seed);runner.alg.policy.reset(torch.ones(raw.num_envs,dtype=torch.bool,device=raw.device))
    obs,_=wrap.get_observations();records={};capture['physics']={};hs=[];cs=[]
    connection.send({'cmd':'reset','num_envs':raw.num_envs});assert connection.recv()['ok']
    initial_motion=arr(cmd.motion_idx)
    assert np.allclose(arr(ball.data.root_pos_w-raw.scene.env_origins)[:,:2],[lateral,-forward],atol=1e-5)
    for step in range(a.steps):
     org=raw.scene.env_origins
     values={'obs':obs,'joint_pos':robot.data.joint_pos,'joint_vel':robot.data.joint_vel,
      'root_pos':robot.data.root_pos_w-org,'root_quat':robot.data.root_quat_w,
      'root_lin_vel':robot.data.root_lin_vel_w,'root_ang_vel':robot.data.root_ang_vel_w,
      'body_pos':robot.data.body_pos_w-org[:,None,:],'body_quat':robot.data.body_quat_w,
      'ref_joint_pos':cmd.joint_pos,'ref_joint_vel':cmd.joint_vel,'ref_ang_vel':cmd.anchor_ang_vel_w,
      'ref_body_pos':cmd.body_pos_w-org[:,None,:], 'ref_body_quat':cmd.body_quat_w,
      'ball_pos':ball.data.root_pos_w-org,'ball_vel':ball.data.root_lin_vel_w,'ball_ang_vel':ball.data.root_ang_vel_w,
      'goal_pos':cmd.target_destination_pos,'last_action':raw.action_manager.action,'applied_torque':robot.data.applied_torque,
      'motion_frame':cmd.time_steps,'motion_index':cmd.motion_idx}
     for k,v in values.items():add(records,k,v)
     with torch.inference_mode():
      wire_obs=arr(obs).astype(np.float32,copy=False)
      connection.send({'cmd':'step','obs_bytes':wire_obs.tobytes(),'shape':wire_obs.shape})
      response=connection.recv();hs.append(np.frombuffer(response['h_pre_bytes'],dtype=np.float32).reshape(response['state_shape']).copy());cs.append(np.frombuffer(response['c_pre_bytes'],dtype=np.float32).reshape(response['state_shape']).copy())
      action_array=np.frombuffer(response['actions_bytes'],dtype=np.float32).reshape(response['actions_shape']).copy()
      actions=torch.from_numpy(action_array).to(raw.device);add(records,'actions',actions)
      capture['active']=True
      obs,_,dones,_=wrap.step(actions)
      capture['active']=False
      if torch.any(dones):raise RuntimeError('Unexpected automatic reset in fixed-horizon capture')
     if step%100==0:print(tag,step,flush=True)
    control={k:np.stack(v) for k,v in records.items()};physics={k:np.stack(v) for k,v in capture['physics'].items()}
    assert physics['ball_pos'].shape[0]==a.steps*4
    np.savez_compressed(O/(tag+'.npz'),**control)
    np.savez_compressed(O/(tag+'_native_states.npz'),h=np.stack(hs),c=np.stack(cs))
    np.savez_compressed(O/(tag+'_physics.npz'),**physics)
    batch={'tag':tag,'forward':forward,'lateral':lateral,'condition':condition,'seed':seed,'envs':raw.num_envs,'steps':a.steps,'initial_motion_index':initial_motion.tolist()}
    batches.append(batch)
    (O/'status.json').write_text(json.dumps({'state':'capturing','completed_batches':len(batches),'total_batches':8},indent=2))
    (O/'batches.json').write_text(json.dumps(batches,indent=2))
  payload={'state':'completed','batches':batches,'checkpoint':str(CHECKPOINT),'checkpoint_sha256':hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
   'actor_backend':'onnx_cpu_shared','strict_fp32':a.strict_fp32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'control_dt':raw.step_dt,'physics_dt':raw.physics_dt,'observation_noise':False,'no_resets':True,
   'reference_body_names':cmd.cfg.body_names,'joint_names':robot.joint_names,'body_names':robot.body_names,
   'scope':'Paired diagnostic, selected positions, clean observations; initial states and reference commands captured from Isaac rather than merely sharing RNG seeds.'}
  (O/f'manifest_{a.batch_index}.json').write_text(json.dumps(payload,indent=2));(O/'status.json').write_text(json.dumps({'state':'completed','completed_batches':8,'total_batches':8}))
 finally:
  connection.close()
  env.close()
try:main()
except BaseException as error:
 import traceback
 (O/f'failure_{a.batch_index}.json').write_text(json.dumps({'error':repr(error),'traceback':traceback.format_exc()},indent=2))
 raise
finally:simulation_app.close()
