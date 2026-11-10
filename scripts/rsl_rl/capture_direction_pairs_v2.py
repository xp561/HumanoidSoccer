"""Exact-state Isaac capture with explicit shared actor observation semantics."""
import argparse,sys
from pathlib import Path
R=Path('/home/tl/Downloads/HumanoidSoccer');sys.path.insert(0,str(R));sys.path.insert(0,str(R/'scripts/rsl_rl'))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--actor-socket',required=True);p.add_argument('--output',required=True);p.add_argument('--num_envs',type=int,default=16);p.add_argument('--smoke-only',action='store_true');AppLauncher.add_app_launcher_args(p);a=p.parse_args();launcher=AppLauncher(a);simulation_app=launcher.app
import json,pickle,hashlib
import numpy as np
import torch,gymnasium as gym
import soccer.tasks
from multiprocessing.connection import Client
from exp.r1_sim2sim.interface import observation
from exp.r1_sim2sim.protocol import noisy_observation
from exp.r1_transfer.ball_observation_guard import bounded_ball_observation
O=Path(a.output);M=json.loads((O/'manifest.json').read_text());B=R/'logs/rsl_rl/r1_flat/2026-10-16_21-27-44_r1_init_curriculum_20261016'
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
 connection=Client(a.actor_socket,family='AF_UNIX',authkey=b'r1-paired-diagnostic');capture={'active':False};original=raw.scene.update
 def update(dt):
  original(dt)
  if capture['active']:
   org=raw.scene.env_origins
   for k,v in {'ball_pos':ball.data.root_pos_w-org,'ball_vel':ball.data.root_lin_vel_w,'foot_pos':robot.data.body_pos_w[:,feet]-org[:,None,:],'contact_force':sensor.data.force_matrix_w}.items():capture.setdefault(k,[]).append(arr(v))
 raw.scene.update=update
 gates=[];items=M['items'][:2] if a.smoke_only else M['items']
 try:
  for start in range(0,len(items),a.num_envs):
   batch=items[start:start+a.num_envs];padded=batch+[batch[-1]]*(a.num_envs-len(batch));packs=[dict(np.load(O/'inputs'/f"case_{x['case']['case']:05d}.npz")) for x in padded]
   env.reset(seed=20261025+start);raw.action_manager.reset();origins=arr(raw.scene.env_origins)
   root=np.stack([np.r_[d['initial_root_pos'],d['initial_root_quat'],d['initial_root_lin_vel'],d['initial_root_ang_vel']] for d in packs]).astype(np.float32);root[:,:3]+=origins
   bstate=np.stack([np.r_[d['initial_ball_pos'],[1.,0.,0.,0.],d['initial_ball_vel'],[0.,0.,0.]] for d in packs]).astype(np.float32);bstate[:,:3]+=origins
   tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=raw.device)
   robot.write_joint_state_to_sim(tensor(np.stack([d['initial_joint_pos'] for d in packs])),tensor(np.stack([d['initial_joint_vel'] for d in packs])))
   robot.write_root_state_to_sim(tensor(root));ball.write_root_state_to_sim(tensor(bstate));raw.scene.reset();raw.sim.forward();raw.scene.update(0.)
   cmd.motion_idx[:]=torch.tensor([cmd.motion.motion_name.index(x['expected']['motion']) for x in padded],device=raw.device);cmd.motion_length[:]=cmd.motion.file_lengths[cmd.motion_idx];cmd.time_steps[:]=0;cmd.target_destination_pos[:]=tensor(np.tile([0.,-5.,.11],(a.num_envs,1)));cmd._update_motion_alignment()
   rng=[np.random.default_rng(np.random.SeedSequence([x['case']['seed'],1])) for x in padded]
   connection.send({'cmd':'reset','num_envs':a.num_envs});assert connection.recv()['ok']
   records={};capture.clear();capture['active']=False
   for step in range(400):
    state={'root_pos':arr(robot.data.root_pos_w)-origins,'root_quat':arr(robot.data.root_quat_w),'root_ang_vel':arr(robot.data.root_ang_vel_w),'joint_pos':arr(robot.data.joint_pos),'joint_vel':arr(robot.data.joint_vel),'ball_pos':arr(ball.data.root_pos_w)-origins}
    last=arr(raw.action_manager.action);inputs=[];cleanobs=[]
    for e,d in enumerate(packs):
     clean=observation(meta,**{k:v[e] for k,v in state.items()},ref_joint_pos=d['ref_joint_pos'][step],ref_joint_vel=d['ref_joint_vel'][step],ref_ang_vel=d['ref_ang_vel'][step],last_action=last[e],goal_pos=[0.,-5.,.11]);cleanobs.append(clean)
     noisy=noisy_observation(clean,meta,runtime,rng[e]);bounded,_,_=bounded_ball_observation(noisy,meta,3.);inputs.append(bounded)
    inputs=np.stack(inputs)
    if step==0:
     body=arr(robot.data.body_pos_w)-origins[:,None,:]
     for e,x in enumerate(batch):
      d=packs[e];errors={'root_position_m':float(np.max(abs(state['root_pos'][e]-d['initial_root_pos']))),'root_quaternion':float(min(np.max(abs(state['root_quat'][e]-d['initial_root_quat'])),np.max(abs(state['root_quat'][e]+d['initial_root_quat'])))),'root_com_velocity_mps':float(np.max(abs(arr(robot.data.root_lin_vel_w)[e]-d['initial_root_lin_vel']))),'root_angular_velocity_radps':float(np.max(abs(state['root_ang_vel'][e]-d['initial_root_ang_vel']))),'ball_velocity_mps':float(np.max(abs(arr(ball.data.root_lin_vel_w)[e]-d['initial_ball_vel']))),'ball_position_m':float(np.max(abs(state['ball_pos'][e]-d['initial_ball_pos']))),'joint_position_rad':float(np.max(abs(state['joint_pos'][e]-d['initial_joint_pos']))),'joint_velocity_radps':float(np.max(abs(state['joint_vel'][e]-d['initial_joint_vel']))),'body_fk_position_m':float(np.max(abs(body[e]-d['initial_body_pos']))),'clean_observation':float(np.max(abs(cleanobs[e]-d['initial_clean_obs']))),'actor_input':float(np.max(abs(inputs[e]-d['policy_inputs'][0])))}
      assert all(v<({'root_position_m':2e-5,'body_fk_position_m':2e-5,'ball_position_m':2e-5,'clean_observation':5e-5,'actor_input':5e-5}.get(k,2e-6)) for k,v in errors.items()),str(errors)
      gates.append({'case':x['case']['case'],'errors':errors})
     write(O/'isaac/initial_gates.json',{'passed':True,'cases':gates,'tolerances':{'position_m':2e-5,'joints':2e-6,'observation':5e-5}})
    connection.send({'cmd':'step','obs_bytes':inputs.astype(np.float32).tobytes(),'shape':inputs.shape});response=connection.recv();actions=np.frombuffer(response['actions_bytes'],dtype=np.float32).reshape(response['actions_shape']).copy()
    for k,v in {**state,'actor_inputs':inputs,'actions':actions}.items():records.setdefault(k,[]).append(v.copy())
    capture['active']=True
    _,_,terminated,truncated,_=env.step(tensor(actions))
    capture['active']=False
    assert not (terminated.any() or truncated.any()),'Unexpected reset'
    expected=np.asarray(meta['action_offset'])+np.asarray(meta['action_scale'])*actions
    error=float(np.max(abs(arr(robot.data.joint_pos_target)[:,action_ids]-expected)))
    assert error<2e-5,'Action target mismatch '+str(error)
    for k,v in {'post_root_pos':robot.data.root_pos_w-raw.scene.env_origins,'post_root_quat':robot.data.root_quat_w,'post_body_pos':robot.data.body_pos_w-raw.scene.env_origins[:,None,:]}.items():records.setdefault(k,[]).append(arr(v))
   data={k:np.stack(v) for k,v in records.items()};physics={k:np.stack(v) for k,v in capture.items() if k!='active'};assert len(physics['ball_pos'])==1600
   for e,x in enumerate(batch):
    idx=x['case']['case'];dest=O/'isaac'/f'case_{idx:05d}.npz';dest.parent.mkdir(exist_ok=True)
    np.savez_compressed(dest,**{k:v[:,e] for k,v in data.items()},**{'physics_'+k:v[:,e] for k,v in physics.items()})
   write(O/'isaac/status.json',{'state':'running','completed_cases':start+len(batch),'total_cases':len(items),'action_target_max_error_last_step':error});print('direction pairs captured',start+len(batch),'/',len(items),flush=True)
  write(O/'isaac/status.json',{'state':'completed','completed_cases':len(items),'total_cases':len(items)})
 finally:connection.close();env.close()
try:main()
except BaseException as e:
 import traceback
 write(O/'isaac/failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
finally:simulation_app.close()
