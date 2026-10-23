"""Predeclared paired pilot; deliberately smaller than the formal grid."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from .aligned import rollout
from .interface import RecurrentPolicy
from .protocol import validate_reference
import numpy as np


def main(artifacts, output):
    runtime=json.loads((output/"runtime_protocol.json").read_text())
    validation=validate_reference(runtime,np.load(output/"protocol_reference.npz"))
    (output/"protocol_validation.json").write_text(json.dumps(validation,indent=2))
    if not validation["passed"]:raise RuntimeError("Protocol validation failed")
    models={"prior_ball_ground":artifacts/"split_ball_ground.xml",
            "probe_contact_candidate":output/"contact_candidate.xml"}
    cases=[]
    for position,(forward,lateral) in enumerate([(1.5,0.),(.5,.25),(1.75,1.),(1.,-.5)]):
        for condition,speeds in [("static",(0.,0.)),("rolling",(.1,.3))]:
            for repetition in range(6):
                cases.append({"forward":forward,"lateral":lateral,"condition":condition,
                              "speed_range":speeds,"seed":58000+100*position+10*(condition=="rolling")+repetition})
    manifest={"scope":"paired pilot, not formal grid", "radius":.5,"steps":400,
              "noise":"captured task observation noise enabled", "cases":cases,
              "models":{k:{"path":str(v),"sha256":hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in models.items()},
              "total_trials":len(models)*len(cases),"started_unix":time.time()}
    (output/"pilot_manifest.json").write_text(json.dumps(manifest,indent=2))
    metadata=json.loads((artifacts/"metadata.json").read_text())
    actor=RecurrentPolicy(artifacts/"policy.onnx",metadata)
    results=[]
    for model_name,model in models.items():
        for i,case in enumerate(cases):
            path=output/"pilot_trials"/model_name/f"trial_{i:03d}.json"
            result=rollout(Path.cwd(),artifacts,runtime,model,case["forward"],case["lateral"],case["seed"],
                           case["speed_range"],.5,path,actor=actor,save_trace=i in [0,6,24,30])
            result.update(model_name=model_name,condition=case["condition"],case=i)
            results.append(result)
            (output/"pilot_status.json").write_text(json.dumps({"status":"RUNNING","completed":len(results),"total":len(cases)*len(models)}))
            print(json.dumps({"completed":len(results),"model":model_name,"goal":result["goal"],"stop":result["stop_reason"]}),flush=True)
    summary={}
    for name in models:
        summary[name]={}
        for condition in ["all","static","rolling"]:
            selected=[r for r in results if r["model_name"]==name and (condition=="all" or r["condition"]==condition)]
            summary[name][condition]={"trials":len(selected),"goals":sum(r["goal"] for r in selected),
                "contacts":sum(r["contacted"] for r in selected),"stops":dict(Counter(r["stop_reason"] for r in selected))}
    (output/"pilot_results.json").write_text(json.dumps({"summary":summary,"trials":results},indent=2))
    (output/"pilot_status.json").write_text(json.dumps({"status":"COMPLETED","completed":len(results),"total":len(results)}))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--artifacts",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();main(args.artifacts,args.output)
