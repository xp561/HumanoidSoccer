"""Contact chronology and paired radius/mass effects, with explicit denominators."""

import argparse
import json
from pathlib import Path

import numpy as np


def episodes(samples, column, threshold=1.0, gap=0.05):
    hits = np.flatnonzero((samples[:, column] > threshold) & (samples[:, 1] > 0))
    if not len(hits):
        return []
    groups = np.split(hits, np.where(np.diff(samples[hits, 0]) > gap)[0] + 1)
    result = []
    for ids in groups:
        start, end = int(ids[0]), int(ids[-1])
        result.append(
            dict(
                start_s=float(samples[start, 0]),
                end_s=float(samples[end, 0]),
                peak_force_n=float(samples[ids, column].max()),
                normal_impulse_ns=float(samples[ids, column].sum() * 0.001),
                ball_velocity_before=samples[max(start - 1, 0), 5:8].tolist(),
                ball_velocity_after=samples[
                    min(end + 1, len(samples) - 1), 5:8
                ].tolist(),
            )
        )
    return result


def analyze_case(path):
    report = json.loads((path / "trial_0001.json").read_text())
    arena = json.loads((path / "arena_events.json").read_text())
    metadata = json.loads((path / "diagnostics/prepared_state.json").read_text())
    with np.load(path / "diagnostics/contact_trace.npz") as archive:
        samples, controls = archive["samples"], archive["controls"]
    active = samples[(samples[:, 1] > 0) & (samples[:, 0] <= 8.000001)]
    policy_control = controls[controls[:, 1] > 0]
    all_events = arena["events"]
    return dict(
        case=path.name,
        radius_m=metadata["radius"],
        mass_kg=metadata["mass"],
        recovered=report["recovered"],
        outcome=report["outcome"],
        policy_commands=report["policy_commands"],
        late_commands=report["policy_commands_after_terminal"],
        policy_observed_s=float(active[-1, 0]),
        peak_ball_speed_mps=float(np.linalg.norm(active[:, 5:8], axis=1).max()),
        maximum_forward_speed_mps=float(active[:, 5].max()),
        maximum_backward_speed_mps=float(active[:, 5].min()),
        final_ball_forward_position_m=float(active[-1, 2]),
        max_policy_tilt_rad=float(policy_control[:, 8].max()),
        max_recorded_tilt_rad=float(controls[:, 8].max()),
        goal_before_8s=any(
            e["kind"] == "goal" and e["sample_end_seconds"] <= 8.000001
            for e in all_events
        ),
        arena_events=all_events,
        left_contacts=episodes(samples, 8),
        right_contacts=episodes(samples, 9),
        initial_robot_max_difference=metadata["robot_initialization_max_difference"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    rows = [
        analyze_case(p)
        for p in sorted((root / "paired").iterdir())
        if (p / "result.json").exists()
    ]
    assert len(rows) == 8
    assert all(r["policy_commands"] == 400 and r["late_commands"] == 0 for r in rows)
    assert all(r["initial_robot_max_difference"] == 0 for r in rows)
    repeats = []
    for variant in ["r070_m205", "r070_m400", "r110_m205", "r110_m400"]:
        a, b = [root / "paired" / f"capture_01_{variant}_repeat{i}" for i in (1, 2)]
        with (
            np.load(a / "diagnostics/contact_trace.npz") as aa,
            np.load(b / "diagnostics/contact_trace.npz") as bb,
        ):
            differences = {
                key: float(np.max(np.abs(aa[key] - bb[key])))
                for key in ("samples", "controls", "commands")
            }
        assert max(differences.values()) < 1e-10, (variant, differences)
        repeats.append(dict(variant=variant, maximum_repeat_differences=differences))
    # Compare instrumented native capture to the previously delivered failing clip.
    older = (
        root.parent
        / "r1_match_arena_20261027/native_small_ball_repeat_02/trial_0001.json"
    )
    original = json.loads(older.read_text())["trace"]
    captured = json.loads((root / "capture_01/trial_0001.json").read_text())["trace"]
    old = [p for p in original if p["stage"] == "kicking"]
    new = [p for p in captured if p["stage"] == "kicking"]
    native_differences = {
        key: float(
            np.max(
                np.abs(
                    np.array([p[key] for p in old]) - np.array([p[key] for p in new])
                )
            )
        )
        for key in ("root_pos", "root_quat", "joint_pos", "ball_pos", "target")
    }
    # Fresh paired snapshot replay should reproduce the native action itself.
    with (
        np.load(root / "capture_01/diagnostics/contact_trace.npz") as a,
        np.load(
            root / "paired/capture_01_r070_m205_repeat1/diagnostics/contact_trace.npz"
        ) as b,
    ):
        native_active, paired_active = a["commands"][:400], b["commands"][:400]
        native_replay_target_max_difference = float(
            np.max(np.abs(native_active - paired_active))
        )

    unique = {
        f"r{round(r['radius_m'] * 1000):03}_m{round(r['mass_kg'] * 1000):03}": r
        for r in rows
        if r["case"].endswith("repeat1")
    }
    y = {k: v["peak_ball_speed_mps"] for k, v in unique.items()}
    effects = dict(
        radius_reduction_at_mass_040=y["r070_m400"] - y["r110_m400"],
        radius_reduction_at_mass_0205=y["r070_m205"] - y["r110_m205"],
        mass_reduction_at_radius_011=y["r110_m205"] - y["r110_m400"],
        mass_reduction_at_radius_007=y["r070_m205"] - y["r070_m400"],
        interaction=y["r070_m205"] - y["r070_m400"] - y["r110_m205"] + y["r110_m400"],
    )
    summary = dict(
        paired_cases=rows,
        exact_repeats=repeats,
        original_failing_clip_max_differences=native_differences,
        native_to_paired_policy_target_max_difference=native_replay_target_max_difference,
        peak_speed_conditional_effects_mps=effects,
        independent_prepared_states=1,
        scope="one prepared state, 2x2 radius/mass factorial, duplicate deterministic replays; not independent success-rate samples",
    )
    (root / "analysis.json").write_text(json.dumps(summary, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(root / "capture_01/diagnostics/contact_trace.npz") as d:
        s, c = d["samples"], d["controls"]
    fig, axes = plt.subplots(
        4, 1, figsize=(11, 10), sharex=True, constrained_layout=True
    )
    axes[0].plot(s[:, 0], s[:, 8], label="Left foot", linewidth=0.8)
    axes[0].plot(s[:, 0], s[:, 9], label="Right foot", linewidth=0.8)
    axes[0].set_ylabel("Ball contact (N)")
    axes[0].legend(loc="upper right")
    axes[1].plot(c[:, 0], c[:, 4], label="Left ankle z")
    axes[1].plot(c[:, 0], c[:, 7], label="Right ankle z")
    axes[1].set_ylabel("Ankle height (m)")
    axes[1].legend(loc="upper right")
    axes[2].plot(s[:, 0], s[:, 5], label="Forward velocity")
    axes[2].plot(s[:, 0], s[:, 6], label="Lateral velocity", alpha=0.7)
    axes[2].set_ylabel("Ball velocity (m/s)")
    axes[2].legend(loc="upper right")
    axes[3].plot(c[:, 0], np.degrees(c[:, 8]), label="Body tilt")
    axes[3].axhline(np.degrees(0.8), color="r", linestyle=":", label="Fault envelope")
    axes[3].set_ylabel("Tilt (degrees)")
    axes[3].set_xlabel("Seconds from one policy start")
    axes[3].legend(loc="upper right")
    for ax in axes:
        ax.axvspan(5.18, 8, color="grey", alpha=0.12)
        ax.axvline(8, color="black", linestyle="--", alpha=0.5)
        ax.grid(alpha=0.2)
        ax.set_xlim(0, max(c[-1, 0], 8.6))
    fig.suptitle(
        "One Action, multiple foot-ball contacts | grey: final reference frame held"
    )
    fig.savefig(root / "contact_timeline.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    order = ["r110_m400", "r110_m205", "r070_m400", "r070_m205"]
    labels = ["11 cm / 400 g", "11 cm / 205 g", "7 cm / 400 g", "7 cm / 205 g"]
    colors = ["#277da8" if unique[k]["recovered"] else "#c54b49" for k in order]
    axes[0].bar(labels, [unique[k]["peak_ball_speed_mps"] for k in order], color=colors)
    axes[0].set_ylabel("Peak ball speed in 8 s (m/s)")
    axes[1].bar(
        labels,
        [np.degrees(unique[k]["max_recorded_tilt_rad"]) for k in order],
        color=colors,
    )
    axes[1].set_ylabel("Peak tilt through recovery (degrees)")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=0.2)
    for ax in axes:
        ax.set_xlabel("Ball radius / mass")
    fig.suptitle(
        "Same prepared robot state | blue: recovered, red: fault | 2 exact replays per cell"
    )
    fig.savefig(root / "paired_radius_mass.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            dict(
                cells=[
                    {
                        k: r[k]
                        for k in (
                            "radius_m",
                            "mass_kg",
                            "peak_ball_speed_mps",
                            "recovered",
                            "goal_before_8s",
                            "max_recorded_tilt_rad",
                        )
                    }
                    for r in unique.values()
                ],
                effects=effects,
                native_difference=native_differences,
                replay_difference=native_replay_target_max_difference,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
