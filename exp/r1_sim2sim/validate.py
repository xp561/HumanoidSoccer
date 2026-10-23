"""Validate R1 policy export, observations and imported kinematics before rollout."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from .interface import RecurrentPolicy, observation
from .model import Robot, build_model


def validate(repo, artifacts, ball_mass, sliding_friction):
    meta = json.loads((artifacts/"metadata.json").read_text())
    replay = np.load(artifacts/"isaac_replay.npz")
    actor = RecurrentPolicy(artifacts/"policy.onnx",meta)
    action_error = state_error = observation_error = 0.0
    state_scaled_error = 0.0
    one_step_state_scaled_error = one_step_state_error = one_step_action_error = 0.0
    term_errors = np.zeros(actor.obs_dim)
    for t,x in enumerate(replay["obs"]):
        action = actor.step(x)
        action_error = max(action_error,float(np.max(np.abs(action-replay["actions"][t]))))
        state_error = max(state_error,float(np.max(np.abs(actor.h-replay["h"][t]))),float(np.max(np.abs(actor.c-replay["c"][t]))))
        # Cell-state magnitudes grow over a rollout; compare float32 results
        # with both an absolute floor and a relative tolerance.
        for actual,expected in [(actor.h,replay["h"][t]),(actor.c,replay["c"][t])]:
            state_scaled_error=max(state_scaled_error,float(np.max(np.abs(actual-expected)/(2e-5+1e-5*np.abs(expected)))))
        previous_h = replay["h"][t-1] if t else np.zeros_like(replay["h"][0])
        previous_c = replay["c"][t-1] if t else np.zeros_like(replay["c"][0])
        direct_action,direct_h,direct_c=actor.session.run(["actions","h_out","c_out"],
            {"obs":x[None],"h_in":previous_h,"c_in":previous_c})
        one_step_action_error=max(one_step_action_error,float(np.max(np.abs(direct_action[0]-replay["actions"][t]))))
        for actual,expected in [(direct_h,replay["h"][t]),(direct_c,replay["c"][t])]:
            one_step_state_error=max(one_step_state_error,float(np.max(np.abs(actual-expected))))
            one_step_state_scaled_error=max(one_step_state_scaled_error,float(np.max(np.abs(actual-expected)/(2e-5+1e-5*np.abs(expected)))))
        obs = observation(meta,**{key:replay[key][t] for key in ["ref_joint_pos","ref_joint_vel","ref_ang_vel","root_pos","root_quat","root_ang_vel","joint_pos","joint_vel","last_action","ball_pos","goal_pos"]})
        term_errors = np.maximum(term_errors,np.abs(obs-x))
    observation_error = float(term_errors.max())
    path = build_model(repo,artifacts,meta,ball_mass=ball_mass,sliding_friction=sliding_friction)
    robot = Robot(path,meta)
    position_error = rotation_error = 0.0
    velocity_error = 0.0
    for t in range(len(replay["obs"])):
        robot.set_state(**{key:replay[key][t] for key in ["root_pos","root_quat","joint_pos","joint_vel","root_lin_vel","root_ang_vel","ball_pos"]})
        velocity=np.empty(6)
        mujoco.mj_objectVelocity(robot.model,robot.data,mujoco.mjtObj.mjOBJ_BODY,robot.root_id,velocity,0)
        velocity_error=max(velocity_error,float(np.max(np.abs(velocity[:3]-replay["root_ang_vel"][t]))),
                           float(np.max(np.abs(velocity[3:]-replay["root_lin_vel"][t]))))
        position_error = max(position_error,float(np.linalg.norm(robot.data.xpos[robot.body_ids]-replay["body_pos"][t],axis=-1).max()))
        dot = np.sum(robot.data.xquat[robot.body_ids]*replay["body_quat"][t],axis=-1)
        rotation_error = max(rotation_error,float(np.max(1-np.abs(dot))))
    mass_error = float(np.max(np.abs(robot.model.body_mass[robot.body_ids]-meta["body_masses"])))
    com_error = float(np.max(np.abs(robot.model.body_ipos[robot.body_ids]-np.array(meta["body_coms"])[:,:3])))
    inertia_error = 0.0
    for i,body in enumerate(robot.body_ids):
        rotation=np.empty(9)
        mujoco.mju_quat2Mat(rotation,robot.model.body_iquat[body])
        rotation=rotation.reshape(3,3)
        local_tensor=rotation@np.diag(robot.model.body_inertia[body])@rotation.T
        inertia_error=max(inertia_error,float(np.max(np.abs(local_tensor-np.asarray(meta["body_inertias"])[i].reshape(3,3)))))
    result = dict(steps=len(replay["obs"]),action_max_abs_error=action_error,cumulative_lstm_state_max_abs_error=state_error,
                  cumulative_lstm_state_max_scaled_error=state_scaled_error,lstm_state_tolerance=dict(atol=2e-5,rtol=1e-5),
                  one_step_action_max_abs_error=one_step_action_error,one_step_lstm_state_max_abs_error=one_step_state_error,
                  one_step_lstm_state_max_scaled_error=one_step_state_scaled_error,
                  recurrent_gate="One-step state equivalence with identical inputs, plus full-sequence action equivalence; cumulative state drift is reported separately",
                  observation_max_abs_error=observation_error,observation_error_by_index=term_errors.tolist(),
                  body_position_max_error_m=position_error,body_quaternion_max_sign_invariant_error=rotation_error,
                  body_mass_max_error_kg=mass_error,local_com_max_error_m=com_error,inertia_tensor_max_error=inertia_error,
                  root_com_velocity_max_error=velocity_error,
                  model=str(path),contact_parameters=dict(ball_mass=ball_mass,sliding_friction=sliding_friction),
                  scope="Interface/kinematic checks; contact dynamics and shooting success are not validated here")
    result["passed"] = all([action_error<2e-4,one_step_action_error<2e-4,one_step_state_scaled_error<=1,observation_error<2e-5,
                             position_error<2e-5,rotation_error<2e-5,mass_error<2e-5,com_error<2e-5,inertia_error<2e-5,velocity_error<2e-5])
    (artifacts/"validation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="observation_error_by_index"},indent=2))
    if not result["passed"]:
        raise RuntimeError("R1 sim2sim interface validation failed; inspect validation.json before rollout")
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",type=Path,required=True)
    p.add_argument("--ball-mass",type=float,required=True)
    p.add_argument("--sliding-friction",type=float,required=True)
    a=p.parse_args()
    validate(Path.cwd(),a.artifacts.resolve(),a.ball_mass,a.sliding_friction)
