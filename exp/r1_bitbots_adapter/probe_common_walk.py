"""Check frozen walker on unchanged kicking geometry before ROS handoff."""
import json
from pathlib import Path
import numpy as np
import mujoco
from .live_worker import LivePlant
from .walking_runtime import WalkingRuntime


def main():
    output = Path('logs/r1_logic_fix/r1_walk_handoff_20261027')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for initial in ('kick_reference', 'walk_default'):
        for command in ([0,0,0],[.2,0,0]):
            p = LivePlant(dict(motion=0,walking=True))
            w = WalkingRuntime()
            r = p.robot
            if initial == 'walk_default':
                order = [p.meta['joint_names'].index(n) for n in w.names]
                r.data.qpos[r.qpos_ids[order]] = w.default_joint_pos
                r.data.qvel[:] = 0
                mujoco.mj_forward(r.model,r.data)
            trace=[]
            for k in range(1000):
                state=p.state()
                target=w.target(joint_names=p.meta['joint_names'],joint_pos=state['joint_pos'],
                                joint_vel=state['joint_vel'],root_quat=state['root_quat'],
                                root_ang_vel=state['root_ang_vel'],command=command if 150<=k<650 else [0,0,0])
                reply=p.step(dict(op='step',joint_names=w.names,positions=target.tolist(),
                                 kp=w.kp.tolist(),kd=w.kd.tolist(),max_torques=w.effort_limit.tolist()))
                d=p.diagnostics();trace.append(dict(seconds=float(r.data.time),**d,root_pos=p.state()['root_pos'].tolist()))
                if d['body_ground_contact'] or d['tilt']>.8 or d['height']<.3:break
            row=dict(initial=initial,command=command,survived=len(trace)==1000,seconds=trace[-1]['seconds'],final=trace[-1],trace=trace)
            rows.append(row)
            print(json.dumps({key:value for key,value in row.items() if key!='trace'}),flush=True)
            (output/'common_model_probe.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
