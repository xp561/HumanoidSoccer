# Live R1 preparation and recovery simulation

Run `bash exp/r1_bitbots_adapter/run_live.sh` from the HumanoidSoccer checkout on china4070.
The launcher uses the existing Bit-Bots default Pixi environment, a separate ROS
domain and localhost discovery. The physical worker uses the original MuJoCo and
policy environment. The native ROS client also requires the fixed-mode and
Action-lifecycle changes in the separate local Bit-Bots workspace; those changes
are not distributed in this repository. Use the configured server workspace for
this integration rather than assuming an unmodified Bit-Bots checkout is sufficient. No dependency declarations or policy weights are changed.
The viewer binds only to loopback. Its startup output gives the URL. Use SSH local
forwarding for the same port when viewing from another computer.

The page shows the live physical camera, robot-forward arrow, requested goal,
control owner, phase, measured support and readiness failures. Start initiates
standstill checking, posture preparation, the fixed-policy ROS Action and recovery.
Cancel stops the Action and attempts recovery; it cannot guarantee balance during
a swing. Reset starts a new physical trial and is not recovery evidence.

The controller never starts real robot drivers, walking, HCM or a get-up controller.
READY returns ownership to a simulation balance/hold controller, not to walking.
The viewer is paced toward wall time; rendering and ROS barriers can slow it down.
Readiness dwell and deadlines are measured in simulation time.

## Preparation and recovery

`balance.py` computes ordinary joint-position targets using measured state, model
Jacobians, mass and gravity, and compressive support forces at points derived from
the foot collision meshes. Targets pass through the ROS ownership router and the
unchanged plant PD gains and effort limits. Planning does not write the simulated
root, joints or ball. Initial placement and the optional test perturbation occur
only when creating a new trial.

The dynamic reference startup frame is not a static standing controller. Feedback
first establishes support, then converges the actuated joints toward the policy
startup posture. Passive head joints remain observed by the actor but are not
preparation targets, because the captured model has no corresponding actuators.
The policy server's actuated-joint startup envelope, yaw/tilt and ball checks remain
in force. Unsupported requests do not run the policy.

Recovery requires uninterrupted fresh physical samples satisfying the posture,
velocity and bilateral load requirements in `ReadinessConfig`. Body-ground contact,
unsafe posture, timeout or a post-terminal policy command prevents READY. The
foot-load test uses actual external normal forces relative to model weight.
The implementation is a local simulation balance controller, not a certified
standing or recovery controller for hardware.

## Acceptance and limitations

Run `python -m exp.r1_bitbots_adapter.validate_live` through the launcher's Pixi
and isolation environment. The output directory contains source-independent state
traces, explicit ownership transitions, actual shot metrics and recovery results.
The supplied cases separately exercise normal startup, reset-only joint deviation,
settled standby, moving-ball rejection, cancellation and deliberately short deadlines.

Successful command handling and physical recovery are separate results. Normal,
perturbed and settled static cases complete the policy and measured recovery.
Preparation cancellation recovers in the tested case. Mid-swing cancellation can
fail recovery and must remain FAULT with HCM ownership; it is not a recovery pass.
The moving-ball preparation case can exceed the existing world-yaw envelope and
is rejected before policy output. This is not evidence of general rolling-ball
support. Arbitrary field heading, walking transitions, get-up and multi-robot play
remain outside the validated scope.

Shot tracking metrics are evaluated only during the policy window. Preparation and
post-policy recovery do not extend the motion-reference clock or produce spurious
tracking failures. Recovery is assessed independently from measured physical state.
No full game success rate is inferred from these deterministic diagnostic cases.
