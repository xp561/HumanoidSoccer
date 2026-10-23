"""Render a saved diagnostic rollout without rerunning or changing its physics."""
import os
os.environ.setdefault("MUJOCO_GL","egl")
import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .model import Robot


def render(artifacts,rollout,output):
    result=json.loads(rollout.read_text());meta=json.loads((artifacts/"metadata.json").read_text())
    trace=np.load(rollout.with_suffix(".npz"))
    tree=ET.parse(result["model"]);root=tree.getroot();world=root.find("worldbody")
    visual=root.find("visual")
    if visual is None:visual=ET.SubElement(root,"visual")
    glob=visual.find("global")
    if glob is None:glob=ET.SubElement(visual,"global")
    glob.set("offwidth","1280");glob.set("offheight","720")
    headlight=visual.find("headlight")
    if headlight is None:headlight=ET.SubElement(visual,"headlight")
    headlight.set("ambient",".5 .5 .5");headlight.set("diffuse",".8 .8 .8")
    ET.SubElement(world,"light",pos="0 -2 5",dir="0 0 -1",diffuse=".8 .8 .8",ambient=".3 .3 .3")
    asset=root.find("asset")
    ET.SubElement(asset,"texture",name="video_checker",type="2d",builtin="checker",rgb1=".35 .4 .35",rgb2=".25 .3 .25",width="512",height="512")
    ET.SubElement(asset,"material",name="video_ground",texture="video_checker",texrepeat="10 10",texuniform="true")
    ground=world.find("geom[@name='ground']");ground.set("material","video_ground");ground.set("rgba","1 1 1 1")
    for name,x,z,size in [("left",-.5,.5,".02 .02 .5"),("right",.5,.5,".02 .02 .5"),("crossbar",0,1,".52 .02 .02")]:
        ET.SubElement(world,"geom",name="video_goal_"+name,type="box",pos=f"{x} -5 {z}",size=size,
                      contype="0",conaffinity="0",rgba=".95 .95 .95 1")
    scene=output.with_suffix(".xml");tree.write(scene)
    robot=Robot(scene,meta);cam=mujoco.MjvCamera()
    cam.lookat[:]=[0,-2,.45];cam.distance=6.5;cam.azimuth=225;cam.elevation=-28
    command=["ffmpeg","-y","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s","1280x720","-r",str(1/meta["control_dt"]),"-i","-",
             "-vf","drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='MuJoCo - R1 learned policy - fixed checkpoint':fontsize=24:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=10:x=20:y=20",
             "-an","-c:v","libx264","-preset","fast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(output)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(robot.model,height=720,width=1280) as renderer:
            for i in range(len(trace["root_pos"])):
                robot.data.qpos[robot.root_q:robot.root_q+3]=trace["root_pos"][i]
                robot.data.qpos[robot.root_q+3:robot.root_q+7]=trace["root_quat"][i]
                robot.data.qpos[robot.qpos_ids]=trace["joint_pos"][i]
                robot.data.qpos[robot.ball_q:robot.ball_q+3]=trace["ball_pos"][i]
                robot.data.qpos[robot.ball_q+3:robot.ball_q+7]=[1,0,0,0]
                mujoco.mj_forward(robot.model,robot.data)
                renderer.update_scene(robot.data,camera=cam)
                process.stdin.write(renderer.render().tobytes())
        process.stdin.close()
        if process.wait():raise RuntimeError("Video encoding failed")
    finally:
        if process.poll() is None:process.terminate();process.wait()
    print(json.dumps({"video":str(output),"frames":len(trace["root_pos"]),"goal":result["goal"],"fell":result["fell"]}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--artifacts",type=Path,required=True);p.add_argument("--rollout",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();render(a.artifacts,a.rollout,a.output)
