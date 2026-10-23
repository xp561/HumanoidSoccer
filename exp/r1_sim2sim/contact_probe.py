"""Fit contact pairs to independent probes and retain held-out errors."""
import argparse
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco


def scene(source, *, impact, damping=.25, time_constant=.005, friction=1.):
    original=ET.parse(source).getroot()
    root=ET.Element("mujoco")
    root.append(copy.deepcopy(original.find("compiler")))
    root.append(copy.deepcopy(original.find("asset")))
    ET.SubElement(root,"option",timestep=".001",integrator="implicitfast",
                  gravity="0 0 0" if impact else "0 0 -9.81",iterations="100")
    world=ET.SubElement(root,"worldbody")
    foot=copy.deepcopy(original.find(".//body[@name='left_ankle_roll_link']"))
    for name in ["quat","axisangle","euler","xyaxes","zaxis"]:foot.attrib.pop(name,None)
    foot.set("pos","0 0 .6" if impact else "3 0 .06")
    for child in list(foot):
        if child.tag in {"joint","freejoint"}:foot.remove(child)
    if not impact:ET.SubElement(foot,"freejoint",name="foot_free")
    world.append(foot)
    # Explicit contact pairs override collision masks. Visual-only meshes must
    # be removed rather than paired alongside their collision duplicates.
    for parent in foot.iter():
        for child in list(parent):
            if child.tag == "geom" and child.get("contype") == "0" and child.get("conaffinity") == "0":
                parent.remove(child)
    geoms=[]
    for i,geom in enumerate(foot.findall(".//geom")):
        geom.set("name",f"foot_geom_{i}");geoms.append(geom)
    contact=ET.SubElement(root,"contact")
    if impact:
        ball=ET.SubElement(world,"body",name="ball",pos=".5 0 .57")
        ET.SubElement(ball,"freejoint",name="ball_free")
        ET.SubElement(ball,"geom",name="ball_geom",type="sphere",size=".11",mass=".4")
        other="ball_geom"
    else:
        ET.SubElement(world,"geom",name="ground",type="plane",size="10 10 .1")
        other="ground"
    for geom in geoms:
        ET.SubElement(contact,"pair",geom1=geom.get("name"),geom2=other,
                      solref=f"{time_constant} {damping}",
                      friction=f"{friction} {friction} .005 .0001 .0001",condim="3")
    return root


def simulate(source, reference, *, impact, damping=.25, time_constant=.005, friction=1.):
    root=scene(source,impact=impact,damping=damping,time_constant=time_constant,friction=friction)
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
    data=mujoco.MjData(model)
    traces=[]
    cases=reference["cases"] if impact else [None]
    for case in cases:
        mujoco.mj_resetData(model,data)
        if impact:
            data.qpos[:3]=case["position"];data.qvel[:3]=case["velocity"]
        else:
            state=np.asarray(reference["slide_initial"])
            data.qpos[:7]=state[:7];data.qvel[:3]=state[7:10]
            # Initial slide angular velocity is explicitly zero in the probe.
            assert np.linalg.norm(state[10:])<1e-12
        mujoco.mj_forward(model,data)
        values=[]
        for i in range(200):
            for _ in range(5):mujoco.mj_step(model,data)
            values.append([data.time,*data.qpos[:3],*data.qvel[:3]])
        traces.append(np.asarray(values))
    return traces


def fit(source, output):
    reference=json.loads((output/"contact_probes_isaac.json").read_text())
    impact_candidates=[]
    for time_constant in [.002,.005,.01,.02]:
        for damping in [.15,.25,.4,.6,1.]:
            traces=simulate(source,reference,impact=True,damping=damping,time_constant=time_constant)
            errors=[float(np.sqrt(np.mean((trace[:,1:4]-np.asarray(case["trace"])[:,1:4])**2)))
                    for trace,case in zip(traces,reference["cases"])]
            impact_candidates.append({"time_constant":time_constant,"damping":damping,
                "position_rmse":errors,"fit_rmse":float(np.mean(np.asarray(errors)[[0,2]])),
                "holdout_rmse":float(np.mean(np.asarray(errors)[[1,3]]))})
    best_impact=min(impact_candidates,key=lambda x:x["fit_rmse"])
    slide_candidates=[]
    expected=np.asarray(reference["slide_trace"])
    for time_constant in [.005,.02]:
        for friction in [.6,.8,1.,1.2]:
            trace=simulate(source,reference,impact=False,friction=friction,damping=1.,time_constant=time_constant)[0]
            slide_candidates.append({"friction":friction,"time_constant":time_constant,
                                     "position_rmse":float(np.sqrt(np.mean((trace[:,1:4]-expected[:,1:4])**2))),
                                     "displacement_x":float(trace[-1,1]-reference["slide_initial"][0])})
    best_slide=min(slide_candidates,key=lambda x:x["position_rmse"])
    baseline=next(c for c in impact_candidates if c["time_constant"]==.02 and c["damping"]==1.)
    root=ET.parse(source).getroot()
    contact=root.find("contact")
    if contact is None:contact=ET.SubElement(root,"contact")
    for body_name in ["left_ankle_roll_link","right_ankle_roll_link"]:
        foot=root.find(f".//body[@name='{body_name}']")
        for i,geom in enumerate(foot.findall(".//geom")):
            if geom.get("contype")=="0" and geom.get("conaffinity")=="0":continue
            if not geom.get("name"):geom.set("name",body_name+"_contact_"+str(i))
            ET.SubElement(contact,"pair",geom1=geom.get("name"),geom2="soccer_ball_geom",
                          solref=f"{best_impact['time_constant']} {best_impact['damping']}",
                          friction="1 1 .005 .0001 .0001",condim="3")
            # Ground and ball pairs are independent; a rebound fit must not
            # turn every robot-floor collision into an underdamped contact.
            f=best_slide["friction"]
            ET.SubElement(contact,"pair",geom1=geom.get("name"),geom2="ground",solref=f"{best_slide['time_constant']} 1",
                          friction=f"{f} {f} .005 .0001 .0001",condim="3")
    candidate=output/"contact_candidate.xml"
    ET.ElementTree(root).write(candidate)
    mujoco.MjModel.from_xml_path(str(candidate))
    result={"impact_fit_cases":["toe_slow","toe_oblique"],"impact_holdout_cases":["toe_fast","side"],
            "impact_candidates":impact_candidates,"impact_baseline":baseline,"impact_best":best_impact,
            "slide_candidates":slide_candidates,"slide_best":best_slide,
            "isaac_slide_displacement_x":float(expected[-1,1]-reference["slide_initial"][0]),
            "candidate_model":str(candidate),
            "limitations":["Only one foot geometry and slide direction tested", "Right-foot parameters transferred by symmetry",
                           "Detached foot probes do not validate full articulated contact response",
                           "Contact sensor aggregation remains simulator-specific"]}
    (output/"contact_fit.json").write_text(json.dumps(result,indent=2))
    traces=simulate(source,reference,impact=True,damping=best_impact["damping"],time_constant=best_impact["time_constant"])
    np.savez(output/"contact_fit_traces.npz",**{c["name"]:trace for c,trace in zip(reference["cases"],traces)})
    print(json.dumps({k:result[k] for k in ["impact_baseline","impact_best","slide_best","isaac_slide_displacement_x"]},indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--model",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();fit(a.model,a.output)
