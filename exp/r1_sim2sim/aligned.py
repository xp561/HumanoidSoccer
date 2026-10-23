"""Bounded MuJoCo evaluation using the captured Isaac evaluation semantics."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import numpy as np
import mujoco
from .interface import RecurrentPolicy, observation
from .model import Robot
from .protocol import noisy_observation, sample_initial, termination_values


def rollout(repo, artifacts, runtime, model_path, forward, lateral, seed,
            speed_range, radius, output, *, steps=400, corruption=True,
            terminations=True, actor=None, save_trace=False):
    metadata = json.loads((artifacts/"metadata.json").read_text())
    runtime = {**runtime, "enable_corruption": corruption and runtime["enable_corruption"]}
    if not json.loads((artifacts/"validation.json").read_text())["passed"]:
        raise RuntimeError("Interface validation failed")
    selected = int(np.argmin(np.linalg.norm(
        np.asarray(metadata["motion_nominal_ball_xy"]) - [lateral, -forward], axis=1)))
    motion_name = metadata["motion_names"][selected]
    with np.load(repo/"motions/r1-corrected"/(motion_name+".npz")) as archive:
        motion = {k: archive[k] for k in archive.files}
    reference_ids = [list(motion["body_names"]).index(n) for n in runtime["reference_body_names"]]
    root_index = list(motion["body_names"]).index("pelvis_link")
    actor = actor or RecurrentPolicy(artifacts/"policy.onnx", metadata)
    actor.reset()
    robot = Robot(model_path, metadata)
    # Independent streams keep physical initial conditions identical in ablations.
    initial_rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    offset, ball_velocity = sample_initial(initial_rng, radius, speed_range)
    initial_root = motion["body_pos_w"][0,root_index].copy()
    initial_root[:2] += offset
    robot.set_state(root_pos=initial_root, root_quat=motion["body_quat_w"][0,root_index],
                    joint_pos=np.clip(motion["joint_pos"][0],
                                      np.asarray(runtime["soft_joint_pos_limits"])[:,0],
                                      np.asarray(runtime["soft_joint_pos_limits"])[:,1]),
                    joint_vel=motion["joint_vel"][0],
                    root_lin_vel=motion["body_lin_vel_w"][0,root_index],
                    root_ang_vel=motion["body_ang_vel_w"][0,root_index],
                    ball_pos=[lateral,-forward,.11], ball_vel=ball_velocity)
    sys.path.insert(0,str(repo/"scripts/rsl_rl"))
    from r1_eval_metrics import KickTrial, goal_plane_normal
    trial = KickTrial()
    goal = np.array([0.,-5.,.11])
    normal = goal_plane_normal(goal[None],np.zeros((1,3)))[0]
    previous = robot.data.qpos[robot.ball_q:robot.ball_q+3].copy()
    last = np.zeros(len(metadata["action_joint_names"]),dtype=np.float32)
    substeps = round(metadata["control_dt"]/robot.model.opt.timestep)
    if abs(substeps*robot.model.opt.timestep-metadata["control_dt"]) > 1e-9:
        raise ValueError("Nonintegral control decimation")
    history_steps = round(runtime["contact_history_length"]*runtime["contact_update_dt"]/robot.model.opt.timestep)
    if history_steps < 1: raise ValueError("Invalid contact history duration")
    contacts = deque([False]*history_steps,maxlen=history_steps)
    ball_geom = robot.model.geom("soccer_ball_geom").id
    traces = {}; stop = "horizon"; errors = {}; flags = {}; first_contact_step = None
    for step in range(steps):
        frame = min(step,len(motion["joint_pos"])-1)
        state = robot.state()
        clean = observation(metadata, **state,
                            ref_joint_pos=motion["joint_pos"][frame],
                            ref_joint_vel=motion["joint_vel"][frame],
                            ref_ang_vel=motion["body_ang_vel_w"][frame,root_index],
                            last_action=last, goal_pos=goal)
        obs = noisy_observation(clean,metadata,runtime,noise_rng)
        action = actor.step(obs); robot.action(action); last = action
        for _ in range(substeps):
            mujoco.mj_step(robot.model,robot.data)
            contacted = False
            for index, contact in enumerate(robot.data.contact):
                other = contact.geom2 if contact.geom1 == ball_geom else contact.geom1 if contact.geom2 == ball_geom else -1
                if other >= 0 and robot.model.geom_bodyid[other] in robot.foot_bodies:
                    force = np.empty(6)
                    mujoco.mj_contactForce(robot.model,robot.data,index,force)
                    contacted |= np.linalg.norm(force[:3]) > 0
            contacts.append(bool(contacted))
        # Refresh Cartesian caches after integration before evaluating body heights.
        mujoco.mj_forward(robot.model,robot.data)
        state = robot.state()
        current = state["ball_pos"]
        velocity = robot.data.qvel[robot.ball_v:robot.ball_v+3].copy()
        flags, errors = termination_values(
            runtime,state["root_pos"],state["root_quat"],robot.data.xpos[robot.body_ids],
            motion["body_pos_w"][frame,reference_ids],motion["body_quat_w"][frame,reference_ids],step+1)
        # Isaac checks terminations before updating the command frame, and its
        # evaluator skips scoring a transition that has already terminated.
        if not np.isfinite(robot.data.qpos).all() or not np.isfinite(robot.data.qvel).all():
            stop = "nonfinite"; trial.finish("episode_end")
        elif terminations and any(flags.values()):
            stop = "+".join(k for k,v in flags.items() if v); trial.finish("episode_end")
        else:
            trial.observe(previous,current,velocity,goal,normal,1.,.5,any(contacts))
            if trial.contacted and first_contact_step is None: first_contact_step = step+1
        previous = current.copy()
        if save_trace:
            for key, value in dict(root_pos=state["root_pos"],root_quat=state["root_quat"],
                                   joint_pos=state["joint_pos"],ball_pos=current,ball_vel=velocity,
                                   action=action,obs=obs,clean_obs=clean).items():
                traces.setdefault(key,[]).append(np.array(value,copy=True))
        if stop != "horizon": break
    result = {"seed":seed,"forward":forward,"lateral":lateral,"motion":motion_name,
              "initial_offset":offset.tolist(),"ball_initial_velocity":ball_velocity.tolist(),
              "radius":radius,"speed_range":list(speed_range),"noise":runtime["enable_corruption"],
              "terminations":terminations,"model":str(model_path),"steps":step+1,
              "seconds":float(robot.data.time),"goal":bool(trial.success),
              "contacted":bool(trial.contacted),"first_contact_step":first_contact_step,
              "end_reason":trial.end_reason,"stop_reason":stop,"terminal_errors":errors,
              "terminal_flags":flags,"max_ball_speed":trial.max_ball_speed,
              "kick_accuracy_cosine":float(trial.kick_accuracy_cosine) if np.isfinite(trial.kick_accuracy_cosine) else None,
              "scope":"bounded protocol pilot; not the full position-grid evaluation"}
    output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(result,indent=2))
    if save_trace: np.savez(output.with_suffix(".npz"),**{k:np.stack(v) for k,v in traces.items()})
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts",type=Path,required=True)
    p.add_argument("--runtime",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--forward",type=float,default=1.5)
    p.add_argument("--lateral",type=float,default=0.)
    p.add_argument("--radius",type=float,default=.5)
    p.add_argument("--speed-range",nargs=2,type=float,default=(0.,0.))
    p.add_argument("--seed",type=int,default=55001)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    print(json.dumps(rollout(Path.cwd(),a.artifacts,json.loads(a.runtime.read_text()),a.model,
                             a.forward,a.lateral,a.seed,a.speed_range,a.radius,a.output,save_trace=True),indent=2))
