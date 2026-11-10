"""Native Bit-Bots BodyDSD and navigation driving the isolated R1 plant."""

import argparse
import base64
import json
import os
import subprocess
import threading
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import (
    PoseStamped,
    PoseWithCovarianceStamped,
    TransformStamped,
    Twist,
)
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from bitbots_body_behavior.body_behavior import BodyDSD
from bitbots_path_planning.path_planning import PathPlanning
from .walking_session import WalkingSession
from .handoff import Phase
from .mujoco_bridge import spin_until
from .supervised_kick import SupervisedKick

HERE = Path(__file__).resolve().parent
BITBOTS = Path("/home/tl/Downloads/r1_ros_integration/bitbots_source")


def flatten(values, prefix=""):
    result = []
    for key, value in values.items():
        name = prefix + str(key)
        if isinstance(value, dict):
            result.extend(flatten(value, name + "."))
        else:
            result.append(Parameter(name, value=value))
    return result


def register_r1_elements(dsd):
    native_actions = dict(dsd.actions)
    native_decisions = dict(dsd.decisions)
    dsd.register_actions(str(HERE / "autonomy_dsd/actions"))
    dsd.actions = {**native_actions, **dsd.actions}
    dsd.register_decisions(str(HERE / "autonomy_dsd/decisions"))
    dsd.decisions = {**native_decisions, **dsd.decisions}


def cancel_abandoned_kick(dsd):
    """Native interrupt replaces the stack without calling action on_pop."""
    capsule = dsd.blackboard.kick
    kick_type = dsd.actions["R1Kick"]
    active = any(isinstance(instance, kick_type) for _, instance in dsd.stack)
    if capsule.is_currently_kicking and not active:
        return capsule.stop_rl_kick()
    return False


def yaw(q):
    return float(
        np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    )


class BehaviorState:
    def __init__(self, node):
        self.node = node
        self.enabled = True
        self.fault = None
        self.status = {}
        self.status_time = -float("inf")
        self.ball_time = -float("inf")
        self.ball_valid = False
        self.ball_stamp = 0
        self.localization_valid = False
        self.completed = 0
        self.search_ticks = 0
        self.approach = {}
        node.create_subscription(String, "motion_status", self.on_status, 10)
        node.create_subscription(
            PoseWithCovarianceStamped, "behavior/ball", self.on_ball, 10
        )

    def on_status(self, msg):
        value = json.loads(msg.data)
        self.status = value
        self.status_time = value["stamp_ns"] / 1e9

    def on_ball(self, msg):
        self.ball_stamp = (
            msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        )
        self.ball_time = self.ball_stamp / 1e9
        p = msg.pose.pose.position
        self.ball_valid = (
            np.isfinite([p.x, p.y, p.z]).all() and msg.header.frame_id == "odom"
        )


class AutonomousSession(WalkingSession):
    walker_domain = 181

    def __init__(
        self,
        output,
        *,
        extra_distance=0.5,
        lateral=0.0,
        reveal_after=4.5,
        record=False,
        scenario="normal",
    ):
        super().__init__(output, realtime=False, render=False)
        self.extra_distance = extra_distance
        self.lateral = lateral
        self.reveal_after = reveal_after
        meta = json.loads(
            Path(
                "logs/r1_logic_fix/r1_mujoco_formal_20261017/artifacts/metadata.json"
            ).read_text()
        )
        with np.load(
            Path("motions/r1-corrected") / (meta["motion_names"][0] + ".npz")
        ) as a:
            q = a["body_quat_w"][0, list(a["body_names"]).index("pelvis_link")]
        self.map_yaw = yaw(q)
        c, s = np.cos(self.map_yaw), np.sin(self.map_yaw)
        self.ball_offset = [
            float(c * extra_distance - s * lateral),
            float(s * extra_distance + c * lateral),
        ]
        self.autonomy_trace = []
        self.navigation_goals = []
        self.velocities = []
        self.last_body_ns = -1
        self.video = None
        self.record_index = 0
        self.record = record
        self.scenario = scenario
        self.injected = False
        self.provider_nodes = []
        self.provider_executor = None
        self.provider_thread = None

    def setup(self, **kwargs):
        super().setup(**kwargs)
        globals = {
            "bot_id": 1,
            "position_number": 0,
            "team_id": 42,
            "role": "offense",
            "field.goal.width": 2.6,
            "field.goal.depth": 0.6,
            "field.size.x": 9.0,
            "field.size.y": 6.0,
            "field.markings.penalty_area.size.x": 1.5,
        }
        globals.update(getattr(self, "field_overrides", {}))
        self.provider_executor = SingleThreadedExecutor()
        for namespace in ("/", "/r1_adapter_test"):
            node = Node(
                "parameter_blackboard",
                namespace=namespace,
                use_global_arguments=False,
                parameter_overrides=[Parameter(k, value=v) for k, v in globals.items()],
                automatically_declare_parameters_from_overrides=True,
            )
            self.provider_nodes.append(node)
            self.provider_executor.add_node(node)
        self.provider_thread = threading.Thread(
            target=self.provider_executor.spin, daemon=True
        )
        self.provider_thread.start()
        self.navigation = PathPlanning()
        self.provider_executor.add_node(self.navigation)
        print("Native navigation initialized; constructing BodyDSD", flush=True)
        config = yaml.safe_load(
            (
                BITBOTS
                / "src/bitbots_behavior/bitbots_body_behavior/config/body_behavior.yaml"
            ).read_text()
        )["body_behavior"]["ros__parameters"]
        animations = yaml.safe_load(
            (
                BITBOTS
                / "src/bitbots_behavior/bitbots_body_behavior/config/animations.yaml"
            ).read_text()
        )["/**"]["ros__parameters"]
        override = yaml.safe_load((HERE / "autonomy.yaml").read_text())["body"]
        config.update(animations)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(config.get(k), dict):
                config[k].update(v)
            else:
                config[k] = v
        config.update(
            use_sim_time=True,
            base_footprint_frame="base_footprint",
            map_frame="map",
            odom_frame="odom",
            dsd_file="main.dsd",
        )
        self.body_node = Node(
            "body_behavior",
            namespace="/r1_adapter_test",
            parameter_overrides=flatten(config),
            automatically_declare_parameters_from_overrides=True,
            cli_args=[
                "--ros-args",
                "-r",
                "rl_kick:=supervised_kick",
                "-r",
                "ball_position_relative_filtered:=behavior/ball",
            ],
        )
        self.body = BodyDSD(self.body_node)
        self.provider_executor.remove_node(self.navigation)
        print("Native BodyDSD initialized", flush=True)
        self.context = BehaviorState(self.body_node)
        self.body.dsd.blackboard.r1 = self.context
        register_r1_elements(self.body.dsd)
        self.body.dsd.load_behavior(str(HERE / "autonomy_dsd/r1_autonomous.dsd"))
        self.supervisor = SupervisedKick(
            self,
            isolated_domain=self.walker_domain,
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        for node in (self.navigation, self.body_node, self.supervisor):
            self.executor.add_node(node)
        self.static_tf = StaticTransformBroadcaster(self.bridge)
        transform = TransformStamped()
        transform.header.frame_id = "map"
        transform.child_frame_id = "odom"
        transform.transform.rotation.w = float(np.cos(-self.map_yaw / 2))
        transform.transform.rotation.z = float(np.sin(-self.map_yaw / 2))
        self.static_tf.sendTransform(transform)
        self.dynamic_tf = TransformBroadcaster(self.bridge)
        self.status_pub = self.bridge.create_publisher(String, "motion_status", 10)
        self.behavior_ball = self.bridge.create_publisher(
            PoseWithCovarianceStamped, "behavior/ball", 10
        )
        self.bridge.create_subscription(PoseStamped, "goal_pose", self.on_goal, 100)
        self.bridge.create_subscription(Twist, "cmd_vel", self.on_velocity, 100)
        self.forget_count = 0

        def reset_filter(request, response):
            self.forget_count += 1
            response.success = True
            response.message = (
                "Simulation observation filter acknowledged; physical ball unchanged"
            )
            return response

        self.bridge.create_service(Trigger, "ball_filter_reset", reset_filter)

        def stop_autonomy(request, response):
            self.manual_stop_requested = True
            self.context.enabled = False
            self.body.dsd.interrupt()
            cancel_abandoned_kick(self.body.dsd)
            self.body.dsd.blackboard.pathfinding.stop_walk()
            response.success = True
            response.message = (
                "Interruption requested; await motion status and terminal Action result"
            )
            return response

        self.bridge.create_service(Trigger, "stop_autonomy", stop_autonomy)
        self.stop_client = self.bridge.create_client(Trigger, "stop_autonomy")
        self.stop_future = None

        if self.record:
            self.video = subprocess.Popen(
                [
                    "/usr/bin/ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "mjpeg",
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(self.output / "autonomous.mp4"),
                ],
                stdin=subprocess.PIPE,
            )
        self.sync(owner=False)

    def on_goal(self, msg):
        self.navigation_goals.append(
            dict(
                seconds=self.seconds,
                frame=msg.header.frame_id,
                x=msg.pose.position.x,
                y=msg.pose.position.y,
            )
        )

    def on_velocity(self, msg):
        self.velocities.append(
            dict(
                seconds=self.seconds,
                vx=msg.linear.x,
                vy=msg.linear.y,
                wz=msg.angular.z,
                stop=msg.angular.x == -1,
            )
        )

    def sync(self, *, owner):
        if not hasattr(self, "status_pub"):
            return super().sync(owner=owner)
        tf = TransformStamped()
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_footprint"
        tf.header.stamp = self.bridge.stamp(self.ns)
        p = tf.transform.translation
        p.x, p.y = map(float, self.state["root_pos"][:2])
        heading = yaw(self.state["root_quat"])
        tf.transform.rotation.w = float(np.cos(heading / 2))
        tf.transform.rotation.z = float(np.sin(heading / 2))
        self.dynamic_tf.sendTransform(tf)
        self.status_pub.publish(
            String(
                data=json.dumps(
                    dict(
                        stamp_ns=self.ns,
                        phase=self.router.gate.phase.value,
                        owner=self.router.gate.owner,
                        outcome=self.outcome,
                    )
                )
            )
        )
        visible = self.seconds >= self.reveal_after and not (
            self.scenario == "ball_loss" and self.seconds >= 6.0
        )
        if visible:
            ball = PoseWithCovarianceStamped()
            ball.header.frame_id = "odom"
            ball.header.stamp = self.bridge.stamp(self.ns)
            (
                ball.pose.pose.position.x,
                ball.pose.pose.position.y,
                ball.pose.pose.position.z,
            ) = map(float, self.state["ball_pos"])
            self.behavior_ball.publish(ball)
        super().sync(owner=owner)
        spin_until(
            self.executor,
            lambda: all(
                n.get_clock().now().nanoseconds == self.ns
                for n in (self.body_node, self.navigation, self.supervisor)
            ),
        )
        spin_until(
            self.executor, lambda: self.context.status.get("stamp_ns") == self.ns
        )
        if visible:
            spin_until(self.executor, lambda: self.context.ball_stamp == self.ns)
        if self.last_body_ns != self.ns:
            self.last_body_ns = self.ns
            try:
                self.body.tf_buffer.lookup_transform("map", "base_footprint", Time())
                self.context.localization_valid = True
            except Exception:
                self.context.localization_valid = False
            before_body_tick = getattr(self, "before_body_tick", None)
            if before_body_tick is not None:
                before_body_tick()
            self.body.loop()
            cancel_abandoned_kick(self.body.dsd)
            for _ in range(5):
                self.executor.spin_once(timeout_sec=0.0001)

    def consume(self, command):
        result = super().consume(command)
        if hasattr(self, "context"):
            self.autonomy_trace.append(
                dict(
                    seconds=self.seconds,
                    owner=self.router.gate.owner,
                    stage=self.stage,
                    root_pos=self.state["root_pos"].tolist(),
                    ball_pos=self.state["ball_pos"].tolist(),
                    diagnostics=self.measured,
                    completed=self.context.completed,
                    approach=self.context.approach.copy(),
                    ball_age=None
                    if not np.isfinite(self.context.ball_time)
                    else self.seconds - self.context.ball_time,
                )
            )
        if (
            self.scenario == "cancel_preparation"
            and self.stage == "preparing"
            and not self.injected
        ):
            self.injected = True
            self.stop_future = self.stop_client.call_async(Trigger.Request())
        if self.video is not None:
            if self.record_index % 5 == 0:
                self.video.stdin.write(
                    base64.b64decode(self.plant.rpc(dict(op="render"))["jpeg"])
                )
            self.record_index += 1
        return result

    def advance(self):
        self.sync(owner=False)
        if self.supervisor.pending is not None:
            self.supervisor.pending = None
            self.shot(preserve_cancel=True)
            self.supervisor.complete(json.loads(Path(self.last_report).read_text()))
            spin_until(self.executor, lambda: not self.supervisor.reserved)
        else:
            if self.unsafe():
                self.fault("unsafe_during_autonomy")
            self.controlled_step(self.router.gate.owner, self.posture)

    def close(self):
        if self.video is not None:
            self.video.stdin.close()
            self.video.wait(timeout=30)
        if self.plant is not None:
            self.plant.close()
        if hasattr(self, "executor"):
            self.executor.shutdown(timeout_sec=3)
            for name in (
                "supervisor",
                "body_node",
                "navigation",
                "walker",
                "bridge",
                "router",
                "adapter",
            ):
                if hasattr(self, name):
                    getattr(self, name).destroy_node()
        if self.provider_executor is not None:
            self.provider_executor.shutdown(timeout_sec=3)
        if self.provider_thread is not None:
            self.provider_thread.join(timeout=3)
        for node in self.provider_nodes:
            node.destroy_node()


def normal_cycle_passed(session, shot, resumed, completed_at):
    """A completed kick must retain healthy walking through the full resume window."""
    return bool(
        session.context.completed >= 1
        and shot
        and shot["recovered"]
        and shot["shot_metrics"]["foot_contact"]
        and shot["policy_commands_after_terminal"] == 0
        and session.context.search_ticks > 0
        and resumed > 0.15
        and completed_at is not None
        and session.seconds - completed_at >= 4.0 - 1e-9
        and session.router.gate.phase == Phase.READY
        and session.router.gate.owner == "walk"
        and session.context.fault is None
        and not session.unsafe()
    )


def main(session_class=AutonomousSession):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-distance", type=float, default=0.5)
    parser.add_argument("--lateral", type=float, default=0.0)
    parser.add_argument("--reveal-after", type=float, default=4.5)
    parser.add_argument("--record", action="store_true")
    parser.add_argument(
        "--scenario",
        choices=["normal", "ball_loss", "cancel_preparation", "recovery_timeout"],
        default="normal",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if os.environ.get("ROS_DOMAIN_ID") != str(session_class.walker_domain):
        raise RuntimeError("Requires isolated simulation domain")
    nav = yaml.safe_load((HERE / "autonomy.yaml").read_text())["navigation"]
    nav.update(getattr(session_class, "navigation_overrides", {}))
    nav["use_sim_time"] = True
    params = args.output / "navigation_parameters.yaml"
    params.write_text(
        yaml.safe_dump(
            {"/r1_adapter_test/bitbots_path_planning": {"ros__parameters": nav}}
        )
    )
    rclpy.init(
        args=[
            "--ros-args",
            "--params-file",
            str(params.resolve()),
            "-r",
            "bitbots_path_planning:__ns:=/r1_adapter_test",
        ]
    )
    session = session_class(
        args.output,
        extra_distance=args.extra_distance,
        lateral=args.lateral,
        reveal_after=args.reveal_after,
        record=args.record,
        scenario=args.scenario,
    )
    try:
        if args.scenario == "recovery_timeout":
            from dataclasses import replace

            session.config = replace(
                session.config, recovery_timeout=session.config.dwell / 5
            )
        session.setup()
        completed_at = None
        resume_position = None
        while session.seconds < 50:
            session.advance()
            if round(session.seconds * 50) % 100 == 0:
                print(
                    json.dumps(
                        dict(
                            seconds=session.seconds,
                            stage=session.stage,
                            completed=session.context.completed,
                            search_ticks=session.context.search_ticks,
                            approach=session.context.approach,
                        )
                    ),
                    flush=True,
                )
            if args.scenario == "ball_loss" and session.seconds >= 12:
                break
            if args.scenario == "cancel_preparation" and session.supervisor.results:
                break
            if session.router.gate.phase.value == "fault" or session.context.fault:
                break
            if session.context.completed and completed_at is None:
                completed_at = session.seconds
                resume_position = session.state["root_pos"].copy()
            if completed_at is not None and session.seconds - completed_at >= 4:
                break
        shot = (
            None
            if session.last_report is None
            else json.loads(Path(session.last_report).read_text())
        )
        resumed = (
            0.0
            if resume_position is None
            else float(
                np.linalg.norm(session.state["root_pos"][:2] - resume_position[:2])
            )
        )
        passed = normal_cycle_passed(session, shot, resumed, completed_at)
        if args.scenario == "ball_loss":
            stale = [x for x in session.autonomy_trace if x["seconds"] > 8.5]
            passed = (
                not session.supervisor.results
                and session.last_report is None
                and not session.navigation.planner.active()
                and bool(stale)
                and np.linalg.norm(
                    np.asarray(stale[-1]["root_pos"])[:2]
                    - np.asarray(stale[0]["root_pos"])[:2]
                )
                / (stale[-1]["seconds"] - stale[0]["seconds"])
                < 0.03
            )
        elif args.scenario == "cancel_preparation":
            passed = (
                bool(session.supervisor.results)
                and session.supervisor.results[-1]["cancelled"]
                and session.stop_future is not None
                and session.stop_future.done()
                and session.stop_future.result().success
                and shot is not None
                and shot["outcome"] == "cancelled"
                and shot["policy_commands"] == 0
                and (
                    shot["recovered"]
                    and shot["final_owner"] == "walk"
                    or not shot["recovered"]
                    and shot["final_owner"] == "hcm"
                )
            )
        elif args.scenario == "recovery_timeout":
            passed = (
                shot is not None
                and shot["outcome"] == "policy_completed"
                and shot["policy_commands"] == 400
                and not shot["recovered"]
                and session.router.gate.owner == "hcm"
                and session.context.completed == 0
            )
        result = dict(
            passed=bool(passed),
            scenario=args.scenario,
            completed=session.context.completed,
            search_ticks=session.context.search_ticks,
            forget_filter_requests=session.forget_count,
            final_stage=session.stage,
            final_owner=session.router.gate.owner,
            resumed_walk_m=resumed,
            resumed_walk_seconds=0.0
            if completed_at is None
            else session.seconds - completed_at,
            final_unsafe=session.unsafe(),
            approach=session.context.approach,
            supervisor_results=session.supervisor.results,
            shot_report=session.last_report,
            shot_metrics=None if shot is None else shot["shot_metrics"],
            observations="MuJoCo ground truth ball and projected pelvis localization, delayed initial visibility",
            physical_resets_after_initialization=0,
            scope="native BodyDSD with R1 tree and native navigation; single robot",
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        (args.output / "trace.json").write_text(json.dumps(session.autonomy_trace))
        (args.output / "navigation.json").write_text(
            json.dumps(
                dict(goals=session.navigation_goals, velocities=session.velocities)
            )
        )
        print(json.dumps(result), flush=True)
        if not result["passed"]:
            raise RuntimeError("Autonomous cycle not completed; inspect evidence")
    finally:
        session.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
