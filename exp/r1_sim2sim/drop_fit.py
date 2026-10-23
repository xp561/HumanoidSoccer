"""Identify nominal contact response against a measured Isaac free-drop trace."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco


def simulate(time_constant,damping_ratio,dt=.001):
    xml=f'''<mujoco><option timestep="{dt}" integrator="implicitfast"/>
    <default><geom friction="1 .005 .0001" solref="{time_constant} {damping_ratio}"/></default>
    <worldbody><geom type="plane" size="5 5 .1"/>
    <body pos="0 0 1.11"><freejoint/><geom type="sphere" size=".11" mass=".4"/></body></worldbody></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);records=[]
    for _ in range(round(1/dt)):
        mujoco.mj_step(model,data);records.append([data.time,data.qpos[2],data.qvel[2]])
    return np.asarray(records)


def fit(path):
    reference=np.array(json.loads((path/"isaac_drop.json").read_text()))
    cases=[]
    for tc in [.005,.01,.02,.04]:
        for damp in [.05,.1,.14,.18,.25,.35,1.]:
            trace=simulate(tc,damp)
            predicted=np.interp(reference[:,0],trace[:,0],trace[:,1])
            error=float(np.sqrt(np.mean((predicted-reference[:,1])**2)))
            i=int(np.argmin(trace[:700,1]));rebound=float(trace[i:,1].max()-.11)
            cases.append(dict(time_constant=tc,damping_ratio=damp,height_rmse_m=error,
                              rebound_height_m=rebound,effective_restitution_from_height=float(np.sqrt(max(0,rebound)))))
    cases.sort(key=lambda x:x["height_rmse_m"])
    best=cases[0]
    np.savez(path/"drop_fit.npz",isaac=reference,mujoco=simulate(best["time_constant"],best["damping_ratio"]))
    result=dict(best=best,all_candidates=cases,scope="Nominal sphere-plane drop only; robot-foot contact and rolling friction remain separate validation tasks")
    (path/"drop_fit.json").write_text(json.dumps(result,indent=2));print(json.dumps(best,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--artifacts",type=Path,required=True)
    fit(p.parse_args().artifacts)
