"""Evaluate the R1 soccer checkpoint in its native Isaac Lab environment."""
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument("--task",default="Tracking-Flat-R1-SoccerMoving-RNN-v0");p.add_argument("--run",default="2026-10-02_02-40-03_r1_first");p.add_argument("--checkpoint",default="model_103998.pt")
p.add_argument("--motion-path",type=Path,default=Path("motions/r1-converted"));p.add_argument("--output-dir",type=Path);p.add_argument("--num-envs",type=int,default=32);p.add_argument("--steps",type=int,default=400);p.add_argument("--seed",type=int,default=42)
p.add_argument("--goal-x",type=float,default=0.0);p.add_argument("--goal-y",type=float,default=-5.0);p.add_argument("--goal-width",type=float,default=1.0);p.add_argument("--contact-speed",type=float,default=.5)
p.add_argument("--ground-label",default="nominal");p.add_argument("--ground-friction",type=float,default=1.0);p.add_argument("--ground-restitution",type=float,default=.8);p.add_argument("--smoke",action="store_true")
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
if a.num_envs<1 or a.steps<1 or not a.motion_path.is_dir():p.error("positive dimensions and a valid motion path are required")
app_launcher=AppLauncher(a)
simulation_app=app_launcher.app
import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
import soccer.tasks
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

@dataclass(frozen=True)
class Scenario:
 name:str;ball_forward:float;ball_lateral:float;ball_speed:float=0.;ball_direction_deg:float=0.;observation_noise:float=0.;initial_yaw_deg:float=0.;perturb_speed:float=0.
def scenarios(smoke):
 b=Scenario("baseline_static",.9,0.)
 if smoke:return [b,Scenario("moving_forward",.9,0.,.2,0.)]
 x=[b]+[Scenario(f"position_f{f:.1f}_l{l:+.1f}",f,l) for f in (.6,.9,1.2) for l in (-.3,0,.3) if(f,l)!=(.9,0)]
 x += [Scenario(f"moving_v{v:.2f}_d{d:+03d}",.9,0.,v,d) for v in (.15,.3) for d in (0,90,180)]
 x += [Scenario(f"obs_noise_{s:.2f}",.9,0.,observation_noise=s) for s in (.05,.1)]
 x += [Scenario(f"yaw_{y:+03d}",.9,0.,initial_yaw_deg=y) for y in (-30,-15,15,30)]
 return x+[Scenario(f"push_{v:.1f}",.9,0.,perturb_speed=v) for v in (.5,1.)]
def crossed(prev,curr,goal,width):
 n=torch.nn.functional.normalize(prev[:,:2]-goal[:,:2],dim=1);ps=((prev[:,:2]-goal[:,:2])*n).sum(1);cs=((curr[:,:2]-goal[:,:2])*n).sum(1);al=torch.where((ps-cs).abs()<1e-9,torch.ones_like(ps),ps/(ps-cs)).clamp(0,1);point=prev[:,:2]+al[:,None]*(curr[:,:2]-prev[:,:2]);tan=torch.stack((-n[:,1],n[:,0]),1);return(ps>0)&(cs<=0)&(((point-goal[:,:2])*tan).sum(1).abs()<=width/2)
def target_slice(raw):
 start=0
 for name,dim in zip(raw.observation_manager.active_terms["policy"],raw.observation_manager.group_obs_term_dim["policy"],strict=True):
  width=int(np.prod(dim))
  if name=="target_point_pos":
   if width!=3:raise RuntimeError(f"target_point_pos width {width}, expected 3")
   return slice(start,start+3)
  start+=width
 raise RuntimeError("target_point_pos missing")
def setup(raw,sc):
 dev,n=raw.device,raw.num_envs;ids=torch.arange(n,device=dev);org=raw.scene.env_origins;local=torch.zeros((n,3),device=dev);local[:,0]=sc.ball_lateral;local[:,1]=-sc.ball_forward;local[:,2]=.11;ang=math.radians(sc.ball_direction_deg);vel=torch.zeros((n,3),device=dev);vel[:,0]=sc.ball_speed*math.sin(ang);vel[:,1]=-sc.ball_speed*math.cos(ang);quat=torch.zeros((n,4),device=dev);quat[:,0]=1
 raw.scene["soccer_ball"].write_root_state_to_sim(torch.cat((local+org,quat,vel,torch.zeros_like(vel)),1),env_ids=ids);term=raw.command_manager.get_term("motion");term.soccer_ball_pos[:]=local;term.target_point_pos[:]=local;term.initial_target_point_pos[:]=local;goal=torch.tensor([a.goal_x,a.goal_y,.11],device=dev).repeat(n,1);term.target_destination_pos[:]=goal
 if sc.initial_yaw_deg:
  root=raw.scene["robot"].data.root_state_w.clone();h=math.radians(sc.initial_yaw_deg)/2;c,s=math.cos(h),math.sin(h);q=root[:,3:7].clone();root[:,3]=c*q[:,0]-s*q[:,3];root[:,4]=c*q[:,1]+s*q[:,2];root[:,5]=c*q[:,2]-s*q[:,1];root[:,6]=c*q[:,3]+s*q[:,0];raw.scene["robot"].write_root_state_to_sim(root)
 return goal+org
def save(path,rows):
 path.mkdir(parents=True,exist_ok=True)
 with (path/"trials.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 groups=defaultdict(list)
 for r in rows:groups[r["scenario"]].append(r)
 out={}
 for k,g in groups.items():
  v=lambda z:np.asarray([r[z] for r in g],float);acc=v("kick_accuracy_cosine");out[k]={"trials":len(g),"success_goal_rate":float(v("success").mean()),"first_contact_rate":float(v("contact").mean()),"kick_accuracy_cosine":float(np.nanmean(acc)) if np.isfinite(acc).any() else None,"kick_accuracy_valid_trials":int(np.isfinite(acc).sum()),"E_gmpjpe_mm":float(v("E_gmpjpe_mm").mean()),"E_mpjpe_mm":float(v("E_mpjpe_mm").mean()),"E_vel":float(v("E_vel").mean())}
 with (path/"summary.json").open("w",encoding="utf-8") as f:json.dump({"config":vars(a),"scenarios":out},f,indent=2,default=str)
def main():
 print('[EVAL] main entered',flush=True)
 torch.manual_seed(a.seed);np.random.seed(a.seed);motions=sorted(a.motion_path.glob("*.npz"))
 if not motions:raise RuntimeError("no R1 motions found")
 cfg=load_cfg_from_registry(a.task,"env_cfg_entry_point");cfg.scene.num_envs=a.num_envs;cfg.commands.motion.motion_files=[str(i.resolve()) for i in motions];cfg.commands.motion.debug_vis=False;cfg.events.push_robot=None;cfg.scene.terrain.physics_material=cfg.scene.terrain.physics_material.replace(static_friction=a.ground_friction,dynamic_friction=a.ground_friction,restitution=a.ground_restitution);cfg.sim.physics_material=cfg.scene.terrain.physics_material
 # These values are consumed by MotionCommand during reset.  This avoids
 # writing PhysX state after reset, which is unsafe in this headless setup.
 initial=scenarios(a.smoke)[0]; angle=math.radians(initial.ball_direction_deg);cfg.commands.motion.eval_fixed_ball_pos=(initial.ball_lateral,-initial.ball_forward,.11);cfg.commands.motion.eval_fixed_ball_lin_vel=(initial.ball_speed*math.sin(angle),-initial.ball_speed*math.cos(angle),0.0);cfg.commands.motion.eval_fixed_target_destination=(a.goal_x,a.goal_y,.11)
 run=Path("logs/rsl_rl/r1_flat")/a.run;ckpt=run/a.checkpoint
 if not ckpt.is_file():raise FileNotFoundError(ckpt)
 env=gym.make(a.task,cfg=cfg);print('[EVAL] environment created',flush=True);wrap=RslRlVecEnvWrapper(env);agent=load_cfg_from_registry(a.task,"rsl_rl_cfg_entry_point");runner=OnPolicyRunner(wrap,agent.to_dict(),log_dir=None,device=wrap.unwrapped.device);runner.load(str(ckpt));print('[EVAL] checkpoint loaded',flush=True);policy=runner.get_inference_policy(device=wrap.unwrapped.device);raw=wrap.unwrapped;print('[EVAL] finding target slice',flush=True);sl=target_slice(raw);print('[EVAL] starting trials',flush=True);rows=[]
 try:
  for sc in scenarios(a.smoke):
   obs,_=wrap.reset();goal=torch.tensor([a.goal_x,a.goal_y,.11],device=raw.device).repeat(raw.num_envs,1)+raw.scene.env_origins;prev=raw.scene["soccer_ball"].data.root_pos_w.clone();succ=torch.zeros(raw.num_envs,dtype=torch.bool,device=raw.device);contact=torch.zeros_like(succ);best=torch.zeros(raw.num_envs,device=raw.device);accuracy=torch.full((raw.num_envs,),float("nan"),device=raw.device);gp=torch.zeros(raw.num_envs,device=raw.device);lp=torch.zeros_like(gp);ve=torch.zeros_like(gp)
   for step in range(a.steps):
    actor=obs["policy"].clone()
    if sc.observation_noise:actor[:,sl]+=torch.randn_like(actor[:,sl])*sc.observation_noise
    with torch.inference_mode():act=policy(actor)
    obs,_,_,_=wrap.step(act);ball=raw.scene["soccer_ball"].data.root_pos_w.clone();vel=raw.scene["soccer_ball"].data.root_lin_vel_w;succ|=crossed(prev,ball,goal,a.goal_width);speed=torch.linalg.vector_norm(vel[:,:2],dim=1);valid=(speed>best)&(speed>=a.contact_speed);direction=torch.nn.functional.normalize(goal[:,:2]-ball[:,:2],dim=1);cos=(vel[:,:2]*direction).sum(1)/speed.clamp_min(1e-6);accuracy[valid]=cos[valid].clamp(-1,1);best=torch.maximum(best,speed);contact|=speed>=a.contact_speed;prev=ball;motion=raw.command_manager.get_term("motion");robot=raw.scene["robot"];actual=robot.data.body_pos_w[:,motion.body_indexes];ref=motion.body_pos_w;gp+=torch.linalg.vector_norm(actual-ref,dim=-1).mean(1);lp+=torch.linalg.vector_norm((actual-actual[:,:1])-(ref-ref[:,:1]),dim=-1).mean(1);ve+=torch.linalg.vector_norm(robot.data.root_lin_vel_w-motion.anchor_lin_vel_w,dim=1)
    if sc.perturb_speed and step==max(1,a.steps//4):root=robot.data.root_state_w.clone();root[:,7]+=sc.perturb_speed;robot.write_root_velocity_to_sim(root[:,7:13])
   for i in range(raw.num_envs):rows.append({"scenario":sc.name,"ground":a.ground_label,**asdict(sc),"trial":i,"success":int(succ[i]),"contact":int(contact[i]),"kick_accuracy_cosine":float(accuracy[i]),"max_ball_speed":float(best[i]),"E_gmpjpe_mm":float(1000*gp[i]/a.steps),"E_mpjpe_mm":float(1000*lp[i]/a.steps),"E_vel":float(ve[i]/a.steps)})
   print(f"[{sc.name}] success={succ.float().mean():.3f} contact={contact.float().mean():.3f}",flush=True)
 finally:env.close()
 out=a.output_dir or run/"evaluation"/f"robustness_{a.ground_label}";save(out,rows);print(f"[PASS] wrote {len(rows)} trials to {out}",flush=True)
try:main()
finally:simulation_app.close()
