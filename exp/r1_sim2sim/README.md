# R1 sim2sim diagnostics

This package builds and checks an R1 MuJoCo scene against runtime data captured
from the matching Isaac Lab checkpoint. It reads the actual named state joints,
action joints, actuator parameters, action scales, defaults, and observation
ordering from the capture metadata. The existing G1 experiment remains separate.

## Artifacts

The artifact directory must contain `metadata.json`, `policy.onnx`, and
`isaac_replay.npz` from a matching checkpoint and motion set. The capture launcher
and its exact command are preserved alongside those files in the experiment log.
Generated URDF/MJCF files and rollout traces belong in that artifact directory.

The ONNX actor includes its observation normalizer and exposes recurrent states
as explicit inputs and outputs. Each diagnostic rollout starts with reset states.
The observation adapter rejects mismatched dimensions or term order.
The floating-root pose is read from the current integrated state. Initial root
linear velocity is converted from Isaac's COM convention to the MuJoCo link
origin convention using the imported COM offset.

## Validation

Use the existing inference environment containing MuJoCo and ONNX Runtime:

```bash
conda run --no-capture-output -n isaaclab python -m exp.r1_sim2sim.validate \
  --artifacts "$R1_ARTIFACTS" \
  --ball-mass "$R1_BALL_MASS" \
  --sliding-friction "$R1_SLIDING_FRICTION"
```

Obtain contact values from the runtime material capture and the actual ball
asset. The output reports action equivalence over the recorded sequence,
single-step recurrent-state equivalence under identical state inputs, cumulative
floating-point state drift, observation reconstruction, forward kinematics,
mass, centers of mass, and complete local inertia tensors. Cumulative recurrent
state drift is reported separately from the single-step operator check.

## Diagnostic rollout

```bash
conda run --no-capture-output -n isaaclab python -m exp.r1_sim2sim.run \
  --artifacts "$R1_ARTIFACTS" --output "$R1_ARTIFACTS/rollout.json"
```

The `--mode` option selects the learned policy, reference PD targets, or holding
the initial joint pose. `--model` selects an independently generated contact-model
variant. `--forward` follows negative world Y and `--lateral` follows positive
world X, matching the Isaac evaluator. The matching reference motion is selected
from the captured nominal ball positions.

Rollouts save observations, actions, joint states, root positions, ball positions,
and ball velocities. Their low-root-height stop condition is a diagnostic guard.
Observation noise and spawn perturbations are disabled in this initial harness.
These runs do not implement the complete formal evaluation protocol and must not
be reported as a paper-comparable success rate.

## Contact identification

`drop_fit` compares nominal sphere-plane drops with `isaac_drop.json` and saves
every candidate and the best matching trajectory. A fitted drop response does
not establish matching robot-foot contacts or rolling resistance. Validate those
separately before expanding to position-grid shooting evaluation.

## Checks

```bash
conda run --no-capture-output -n isaaclab python -m unittest discover \
  -s tests -p test_r1_sim2sim.py -v
```

## Captured protocol and independent probes

The aligned evaluator reads its noise distributions, termination thresholds,
body names, soft joint limits and timing from the runtime protocol capture.
It follows Isaac's post-physics termination ordering and excludes the final
terminated transition from scoring. Initial perturbations use the formal
manifest's uniform-area disk and rolling-ball distributions. Different RNG
implementations match distributions rather than identical random samples.

The contact probe fitter excludes visual-only meshes before creating explicit
contact pairs. Fit and held-out impact cases are declared separately. Ground
and foot-ball pairs remain independent. The pilot records paired initial states,
all outcomes and source hashes; it must not be substituted for the formal grid.
The prototype runner remains available with its original diagnostic settings.

Example aligned rollout:

    conda run --no-capture-output -n isaaclab python -m exp.r1_sim2sim.aligned \
      --artifacts "$R1_ARTIFACTS" --runtime "$R1_RUNTIME_PROTOCOL" \
      --model "$R1_CONTACT_MODEL" --output "$R1_ROLLOUT"

The detached-foot experiments diagnose engine contact response, not articulated
balance. Retain candidate settings as experimental until held-out impacts and
paired shooting support them; remaining force aggregation, collision cooking,
solver and velocity-limit differences must be reported.

## Fixed-horizon task benchmark

The benchmark adapter uses an explicit frozen profile for initialization,
motion selection, contacts and scoring. Fixed world ball positions are kept
independent of robot spawn offsets. Overlapping initial states are rejected.
Nominal motion endpoints are recentered when the profile uses a world-origin
robot spawn disk.

The common scorer retains separate fixed-horizon, foot-contact-gated,
tracking-valid and legacy strict outcomes. Direction cosines use the first
impact point; first-qualified and peak-speed samples remain distinguishable.
Tracking failure does not reset the fixed-horizon simulation. Numerical
failures remain failed attempts in the denominator.

Use the generated formal launcher and manifest for reproducible grid runs.
The manifest freezes policy, source, model, mesh and reference-motion inputs.
Completed trial records support resumption only with unchanged inputs.
Report the core workspace separately from the full heatmap, and declare
operational choices that the publication does not specify. A shared scene
template alone does not establish an identical G1 policy evaluation.
