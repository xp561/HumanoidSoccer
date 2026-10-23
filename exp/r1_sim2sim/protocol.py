"""Evaluation semantics captured from the active Isaac task at runtime."""
import math
import numpy as np
from .interface import inverse_rotate


def sample_initial(rng, radius, speed_range):
    if radius < 0 or not np.isfinite(radius):
        raise ValueError("Invalid initial disk radius")
    low, high = speed_range
    if not 0 <= low <= high or not np.isfinite(high):
        raise ValueError("Invalid ball speed range")
    distance = radius * np.sqrt(rng.random())
    angle = rng.uniform(0, 2*np.pi)
    speed = rng.uniform(low, high)
    direction = rng.uniform(0, 2*np.pi)
    return (np.array([distance*np.cos(angle), distance*np.sin(angle)]),
            np.array([speed*np.cos(direction), speed*np.sin(direction), 0.]))


def noisy_observation(obs, metadata, runtime, rng):
    n = len(metadata["joint_names"])
    a = len(metadata["action_joint_names"])
    sizes = [2*n, 3, 3, 3, n, n, a, 3, 3]
    terms = metadata["observation_terms"]
    if len(terms) != len(sizes):
        raise ValueError("Unsupported observation layout")
    result = np.array(obs, dtype=np.float32, copy=True)
    if result.shape != (sum(sizes),):
        raise ValueError("Observation dimension mismatch")
    if not runtime["enable_corruption"]:
        return result
    offset = 0
    for name, size in zip(terms, sizes):
        cfg = runtime["noise"][name]
        if cfg is not None:
            if cfg["type"] != "UniformNoiseCfg" or cfg["operation"] != "add":
                raise ValueError("Unsupported runtime noise configuration")
            result[offset:offset+size] += rng.uniform(cfg["min"], cfg["max"], size).astype(np.float32)
        offset += size
    return result


def termination_values(runtime, root_pos, root_quat, body_pos,
                       ref_body_pos, ref_body_quat, episode_step):
    names = runtime["reference_body_names"]
    anchor = names.index("pelvis_link")
    robot_names = runtime["robot_body_names"]
    output = {}
    errors = {}
    for name, cfg in runtime["terminations"].items():
        function = cfg["function"]
        params = cfg["params"]
        if function == "time_out":
            output[name] = episode_step >= math.ceil(runtime["episode_length_s"]/runtime["control_dt"])
            continue
        if function == "bad_anchor_pos_z_only":
            error = abs(ref_body_pos[anchor, 2] - root_pos[2])
        elif function == "bad_anchor_ori":
            error = abs(inverse_rotate(ref_body_quat[anchor], [0,0,-1])[2]
                        - inverse_rotate(root_quat, [0,0,-1])[2])
        elif function == "bad_motion_body_pos_z_only":
            # Isaac's alignment applies only yaw and planar translation, so it
            # preserves the reference body's absolute height.
            error = max(abs(ref_body_pos[names.index(body), 2]
                            - body_pos[robot_names.index(body), 2])
                        for body in params["body_names"])
        else:
            raise ValueError("Unsupported runtime termination: " + function)
        errors[name] = float(error)
        output[name] = bool(error > params["threshold"])
    return output, errors


def validate_reference(runtime, trace):
    disagreements = []
    maximum_errors = {}
    total = 0
    for t in range(len(trace["post_root_pos"])):
        for e in range(trace["post_root_pos"].shape[1]):
            flags, errors = termination_values(
                runtime, trace["post_root_pos"][t,e], trace["post_root_quat"][t,e],
                trace["post_body_pos"][t,e], trace["post_ref_body_pos"][t,e],
                trace["post_ref_body_quat"][t,e], int(trace["post_episode_length"][t,e]))
            total += 1
            for name, flag in flags.items():
                expected = bool(trace["post_term_"+name][t,e])
                if flag != expected:
                    disagreements.append({"step": t, "env": e, "term": name,
                                          "expected": expected, "actual": flag})
            for name, error in errors.items():
                maximum_errors[name] = max(maximum_errors.get(name, 0.), error)
    return {"passed": not disagreements, "states": total,
            "disagreements": disagreements, "maximum_observed_errors": maximum_errors,
            "note": "Distribution-equivalent sampling; NumPy and Torch random streams are not bit-identical."}
