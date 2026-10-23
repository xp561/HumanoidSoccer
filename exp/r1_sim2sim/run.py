"""Small diagnostic rollouts; not the formal paper-comparison evaluator."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import mujoco
from .interface import RecurrentPolicy,observation
from .model import Robot


def rollout(repo,artifacts,forward,lateral,steps,mode,ball_speed,seed,output,model_path=None):
    meta=json.loads((artifacts/"metadata.json").read_text())
    validation=json.loads((artifacts/"validation.json").read_text())
    if not validation["passed"]:raise RuntimeError("Interface validation must pass before rollout")
    xy=np.array([lateral,-forward])
    selected=int(np.argmin(np.linalg.norm(np.asarray(meta["motion_nominal_ball_xy"])-xy,axis=1)))
    name=meta["motion_names"][selected]
    motion=np.load(repo/"motions/r1-corrected"/(name+".npz"))
    if abs(float(np.asarray(motion["fps"]).reshape(-1)[0])*meta["control_dt"]-1)>1e-6:
        raise ValueError("Reference frame rate does not match the policy control period")
    model_path=model_path or artifacts/"r1_soccer.xml"
    robot=Robot(model_path,meta)
    actor=RecurrentPolicy(artifacts/"policy.onnx",meta)
    initial=dict(root_pos=motion["body_pos_w"][0,0],root_quat=motion["body_quat_w"][0,0],
                 joint_pos=motion["joint_pos"][0],joint_vel=motion["joint_vel"][0],
                 root_lin_vel=motion["body_lin_vel_w"][0,0],root_ang_vel=motion["body_ang_vel_w"][0,0])
    rng=np.random.default_rng(seed);angle=rng.uniform(0,2*np.pi)
    robot.set_state(**initial,ball_pos=[lateral,-forward,.11],ball_vel=[ball_speed*np.cos(angle),ball_speed*np.sin(angle),0])
    sys.path.insert(0,str(repo/"scripts/rsl_rl"))
    from r1_eval_metrics import KickTrial,goal_plane_normal
    trial=KickTrial();goal=np.array([0,-5,.11]);normal=goal_plane_normal(goal[None],np.zeros((1,3)))[0]
    last=np.zeros(len(meta["action_joint_names"]),dtype=np.float32)
    substeps=round(meta["control_dt"]/robot.model.opt.timestep)
    assert abs(substeps*robot.model.opt.timestep-meta["control_dt"])<1e-9
    records={};previous=robot.data.qpos[robot.ball_q:robot.ball_q+3].copy()
    min_height=10.;max_speed=0.;fell=False;stop="horizon"
    for step in range(steps):
        frame=min(step,len(motion["joint_pos"])-1)
        state=robot.state()
        refq=motion["joint_pos"][frame];refv=motion["joint_vel"][frame]
        obs=observation(meta,**state,ref_joint_pos=refq,ref_joint_vel=refv,
                        ref_ang_vel=motion["body_ang_vel_w"][frame,0],last_action=last,goal_pos=goal)
        if mode=="learned":action=actor.step(obs)
        else:
            target=refq[meta["action_joint_ids"]] if mode=="reference" else initial["joint_pos"][meta["action_joint_ids"]]
            action=(target-np.asarray(meta["action_offset"]))/np.asarray(meta["action_scale"])
        robot.action(action);last=action
        contact=False
        for _ in range(substeps):
            mujoco.mj_step(robot.model,robot.data)
            contact=contact or robot.ball_foot_contact()
        current=robot.data.qpos[robot.ball_q:robot.ball_q+3].copy()
        velocity=robot.data.qvel[robot.ball_v:robot.ball_v+3].copy()
        trial.observe(previous,current,velocity,goal,normal,1.,.5,contact)
        previous=current
        z=float(robot.data.qpos[robot.root_q+2]);min_height=min(min_height,z)
        max_speed=max(max_speed,float(np.max(np.abs(robot.data.qvel[robot.dof_ids]))))
        for key,val in dict(root_pos=robot.data.qpos[robot.root_q:robot.root_q+3].copy(),
                            root_quat=robot.data.qpos[robot.root_q+3:robot.root_q+7].copy(),joint_pos=robot.data.qpos[robot.qpos_ids].copy(),
                            ball_pos=current,ball_vel=velocity,action=action.copy(),obs=obs).items():records.setdefault(key,[]).append(val)
        if not np.isfinite(robot.data.qpos).all() or not np.isfinite(robot.data.qvel).all():
            stop="nonfinite";break
        if z<.3:
            fell=True;stop="low_root_height";break
    result=dict(mode=mode,model=str(model_path),physics_dt=float(robot.model.opt.timestep),forward=forward,lateral=lateral,seed=seed,ball_speed=ball_speed,motion=name,
                completed_steps=len(records["root_pos"]),simulated_seconds=float(robot.data.time),stop_reason=stop,
                fell=fell,min_root_height=min_height,max_joint_speed=max_speed,foot_contact=trial.contacted,
                goal=trial.success,kick_accuracy_cosine=float(trial.kick_accuracy_cosine) if np.isfinite(trial.kick_accuracy_cosine) else None,
                scope="Diagnostic rollout; zero spawn perturbation and observation noise disabled; not a formal success-rate estimate")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    np.savez(output.with_suffix(".npz"),**{k:np.stack(v) for k,v in records.items()})
    print(json.dumps(result,indent=2));return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--artifacts",type=Path,required=True)
    p.add_argument("--forward",type=float,default=1.5);p.add_argument("--lateral",type=float,default=0.)
    p.add_argument("--steps",type=int,default=400);p.add_argument("--mode",choices=["learned","reference","hold"],default="learned")
    p.add_argument("--ball-speed",type=float,default=0.);p.add_argument("--seed",type=int,default=53001)
    p.add_argument("--model",type=Path)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    rollout(Path.cwd(),a.artifacts.resolve(),a.forward,a.lateral,a.steps,a.mode,a.ball_speed,a.seed,a.output,a.model)
