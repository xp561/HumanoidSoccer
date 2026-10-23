"""Replay and closed-loop parity checks for the offline R1 adapter."""
import argparse
import json
from pathlib import Path
import numpy as np
from exp.r1_bitbots_adapter.runtime import Runtime, Reference
from exp.r1_sim2sim.interface import RecurrentPolicy
import exp.r1_sim2sim.benchmark as benchmark_module

ROOT = Path.cwd()
FORMAL = ROOT / "logs/r1_logic_fix/r1_mujoco_formal_20261017"
KEYS = ["ref_joint_pos", "ref_joint_vel", "ref_ang_vel", "root_pos",
        "root_quat", "root_ang_vel", "joint_pos", "joint_vel",
        "last_action", "ball_pos", "goal_pos"]


def replay(output):
    runtime = Runtime(FORMAL / "artifacts")
    old = RecurrentPolicy(FORMAL / "artifacts/policy.onnx", runtime.meta)
    data = np.load(ROOT / "logs/r1_logic_fix/r1_sim2sim_20261017/isaac_replay.npz")
    errors = dict(observation_vs_isaac=0., action_vs_existing_runtime=0.,
                  recurrent_state_vs_existing_runtime=0., action_vs_isaac=0.)
    for i, x in enumerate(data["obs"]):
        obs = runtime.observation(**{key: data[key][i] for key in KEYS})
        errors["observation_vs_isaac"] = max(errors["observation_vs_isaac"], float(np.max(np.abs(obs-x))))
        # Identical observations isolate recurrent inference from observation reconstruction.
        actual = runtime.step(x)
        expected = old.step(x)
        errors["action_vs_existing_runtime"] = max(errors["action_vs_existing_runtime"], float(np.max(np.abs(actual-expected))))
        errors["recurrent_state_vs_existing_runtime"] = max(errors["recurrent_state_vs_existing_runtime"],
            float(np.max(np.abs(runtime.h-old.h))), float(np.max(np.abs(runtime.c-old.c))))
        errors["action_vs_isaac"] = max(errors["action_vs_isaac"], float(np.max(np.abs(actual-data["actions"][i]))))
    result = dict(states=len(data["obs"]), max_abs_error=errors)
    result["passed"] = (
        errors["observation_vs_isaac"] < 2e-5
        and errors["action_vs_existing_runtime"] == 0
        and errors["recurrent_state_vs_existing_runtime"] == 0
        and errors["action_vs_isaac"] < 2e-4)
    (output / "replay.json").write_text(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("Replay parity failed")
    return result


def closed_loop(output, count):
    profile = json.loads((FORMAL / "profile.json").read_text())
    bench = benchmark_module.Benchmark(ROOT, FORMAL / "artifacts",
        FORMAL / "runtime_protocol.json", FORMAL / "model.xml", profile)
    runtime = Runtime(FORMAL / "artifacts")
    refs = [Reference(ROOT / "motions/r1-corrected" / (name+".npz"), runtime)
            for name in runtime.meta["motion_names"]]
    cases = []
    # Fixed previously evaluated cases, spanning center, both sides and range.
    positions = [(0.5, 0.), (1., -1.), (1., 1.), (1.5, 0.)]
    for condition in ("static", "rolling"):
        found = {}
        for path in sorted((FORMAL / "trials" / condition).glob("*.json")):
            case = json.loads(path.read_text())
            key = (case["forward"], case["lateral"])
            if key in positions and len(found.get(key, [])) < count:
                found.setdefault(key, []).append(case)
        for key in positions:
            if len(found.get(key, [])) != count:
                raise RuntimeError("Missing frozen case " + str((condition, key)))
            cases.extend(found[key])
    original_observation = benchmark_module.observation
    original_actor = bench.actor
    original_action = bench.robot.action
    results = []
    for index, case in enumerate(cases):
        key = f'{case["condition"]}_{case["case"]:05d}'
        baseline = bench.run(case, output / (key+"_baseline.json"), save_trace=True)
        selection = int(np.argmin(np.linalg.norm(bench.nominal - [case["lateral"], -case["forward"]], axis=1)))
        ref = refs[selection]
        tick = [0]
        def adapter_observation(meta, **kwargs):
            expected = ref.sample(tick[0]*runtime.dt)
            for name in expected:
                np.testing.assert_array_equal(expected[name], kwargs.pop(name))
            tick[0] += 1
            # Exercise a permuted incoming JointState ordering, including head joints.
            q, dq = runtime.named_joints(runtime.names[::-1],
                kwargs["joint_pos"][::-1], kwargs["joint_vel"][::-1])
            kwargs.update(joint_pos=q, joint_vel=dq)
            return runtime.observation(**kwargs, **expected)
        def adapter_target(action):
            target = runtime.targets(action)
            bench.robot.data.ctrl[bench.robot.action_ids] = target.positions
        try:
            bench.actor = runtime
            bench.robot.action = adapter_target
            benchmark_module.observation = adapter_observation
            adapted = bench.run(case, output / (key+"_adapter.json"), save_trace=True)
        finally:
            bench.actor = original_actor
            bench.robot.action = original_action
            benchmark_module.observation = original_observation
        a = np.load(output / (key+"_baseline.npz"))
        b = np.load(output / (key+"_adapter.npz"))
        deltas = {name: float(np.max(np.abs(a[name]-b[name]))) for name in a.files}
        outcomes = ("goal", "kicked_goal", "foot_contact", "tracking_failed",
                    "numerical_failure", "legacy_strict_goal")
        result = dict(case=case["case"], condition=case["condition"],
            forward=case["forward"], lateral=case["lateral"], max_abs_delta=deltas,
            baseline_goal=baseline["kicked_goal"], adapter_goal=adapted["kicked_goal"],
            matches_frozen_result=all(baseline[k] == case[k] for k in outcomes),
            passed=all(delta == 0 for delta in deltas.values())
                and all(baseline[k] == adapted[k] for k in outcomes)
                and not adapted["numerical_failure"])
        results.append(result)
        print(json.dumps(dict(completed=index+1, total=len(cases), **result)), flush=True)
        (output / "closed_loop_progress.json").write_text(json.dumps(results, indent=2))
        if not result["passed"] or not result["matches_frozen_result"]:
            raise RuntimeError("Closed-loop parity failed: " + key)
    return dict(paired_cases=len(results), simulator_runs=2*len(results),
                passed=True, cases=results,
                scope="Interface equivalence on selected frozen cases; not a new success-rate estimate or ROS/hardware test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise ValueError("Repetitions must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(replay=replay(args.output))
    result["closed_loop"] = closed_loop(args.output, args.repetitions)
    result["passed"] = True
    (args.output / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
