# R1 Bit-Bots adapter prototype

Development and validation run on china4070 in the HumanoidSoccer checkout.
The offline core preserves the selected original curriculum policy; weights and rewards
are unchanged. Observation-noise fine-tuning was not promoted.

## Components

- runtime.py: exact exported observations, named-joint sensor ordering, transactional
  recurrent state updates, target positions and metadata gains/effort limits.
- Reference: reference sampling by elapsed simulation time, clamped without wrapping.
- ros_node.py: opt-in isolated ROS Action server using the tested runtime.
- test_runtime.py: malformed inputs, mapping, reset and conversion tests.
- test_ros.py: native generated-message serialization and Action lifecycle tests.
- validate.py: saved-state replay and paired MuJoCo interface comparisons.
- physics_worker.py: original-environment MuJoCo plant with strict target/gain validation.
- mujoco_bridge.py: actual ROS sensor and command serialization with simulation-clock barriers.
- batch_mujoco.py: nominal static/rolling paired comparisons and fault injection.

## ROS prototype boundary

The server is restricted to its test namespace and requires an explicitly selected ROS
domain. It does not start HCM, DSD, walking, simulator or hardware drivers. It has not
been connected to real hardware or to the complete Bit-Bots control stack.

The action is rl_kick, with the original bitbots_msgs/Kick type, and the output is
kick_motor_goals using the original JointCommand type. Both are relative to the isolated
namespace. The original message definitions were built unchanged from the pinned
Bit-Bots source revision recorded in the report.

Only the forward reference-compatible prototype is accepted. The actor has no explicit
ball-speed input: the fixed-policy strength sentinel is enforced in request validation.
Ordinary Bit-Bots speed requests are rejected; supporting them requires an explicit
upper-layer contract or a differently trained policy. Do not silently equate strength
with torque or achieved ball speed.

Inputs:
- joint_states: JointState, all metadata observed joints with finite position and velocity.
- pelvis_odometry: Odometry describing pelvis_link in odom, with angular velocity in the child frame.
- ball_position_relative_filtered: timestamped PoseWithCovarianceStamped containing the ball center;
  TF conversion to odom is performed at the measurement timestamp.
- robot_state: RobotControlState, allowed states and freshness checked before and during execution.
- walking_released: Bool, an explicit test handshake checked before and during execution.

walking_released is supplied by a synthetic source in these tests. It is not proof of
exclusive ownership in the actual HCM. The real walking-release and recovery handshake
remains to be implemented and tested.

Parameters are declared in KickNode; the declarations are authoritative for periods,
limits, paths and task geometry. Command timestamps are current publication times.
Actions are converted with the exported offset and scale; observed joints and commanded
joints use their own metadata mappings. Motor signs, calibration, limits and effort
enforcement on R1 hardware are not validated here.

The server uses asynchronous Action completion and a single-threaded executor. Cancelling,
losing fresh inputs/ownership or entering a disallowed HCM state aborts publication.
Short request timeout is an abort, not a completed execution. SUCCESS only denotes
completion of the configured control interval; it does not mean a detected goal.

## Validate on china4070

```bash
bash /home/tl/Downloads/r1_ros_integration/test_action.sh
```

The launcher uses the remote Bit-Bots default Pixi environment, the original frozen lock
file, a separate ROS domain and localhost discovery. Do not manually source the overlay
inside Pixi. Its paths and environment selections can be inspected in the launcher.

Offline MuJoCo interface checks remain runnable from the remote HumanoidSoccer root:

```bash
/home/tl/miniconda3/envs/isaaclab/bin/python -m unittest exp.r1_bitbots_adapter.test_runtime -v
/home/tl/miniconda3/envs/isaaclab/bin/python -m exp.r1_bitbots_adapter.validate \
  --output logs/r1_logic_fix/r1_bitbots_adapter_20261023
```

Native ROS test evidence is in logs/r1_logic_fix/r1_ros_action_20261023.
Paired MuJoCo evidence is in logs/r1_logic_fix/r1_bitbots_adapter_20261023.
The ROS source fixture uses stationary synthetic states and a command sink; it is not a
closed-loop shooting evaluation. Runtime artificial observation noise is disabled.

## ROS/MuJoCo closed-loop validation

The physical bridge has now been tested through actual ROS messages against a direct
controller. See logs/r1_logic_fix/r1_ros_mujoco_20261023/REPORT.md for measured results,
source hashes, per-case outcomes and scope.

The plant runs in the original inference environment, keeping the frozen MuJoCo version,
model, contact configuration and physical step. It is isolated from the Jazzy interpreter
and libraries. State delivery is acknowledged before advancing the ROS simulation clock.
The first reference action is emitted at Action activation; each following clock interval
produces one timestamp-matched target. Terminal results are followed by an explicit check
that the policy publishes no further command.

The bridge records the actual observations passed into inference and compares full
trajectories, not only goal booleans. A simulator-only current-position fallback is used
after terminal results; it does not implement HCM recovery or prove that motion has stopped.
The lockstep benchmark does not test wall-clock real-time deadlines or camera latency.

Run on china4070:

```bash
bash /home/tl/Downloads/r1_ros_integration/test_mujoco.sh
```

The batch runner recomputes cases by default. Its explicit reuse flag is intended only for
resuming an unchanged source/asset combination. Results include separate static/rolling
nominal cases and cancellation, ball-dropout and ownership-loss injections.

## Next integration work

Link the real walking-release/HCM contract and DSD task handling. Validate preparation,
exclusive ownership, normal and abnormal return, and recovery before integrating an R1
walking controller. The actual R1 walking policy and execution interface must be identified;
the old robot's walking network cannot supply this function directly.

Validate target geometry, dynamic TF, measurement latency and world-yaw coverage before
expanding beyond the forward prototype. Execution completion and physical goal detection
must remain separate.

## Live measured preparation and recovery

The isolated live session now adds feedback preparation, a real-time camera and
measured post-policy readiness. See `LIVE_SIMULATION.md` for startup, controls,
acceptance boundaries and remaining cancellation/rolling-ball limitations. The
older benchmark's synthetic readiness acknowledgements are not used as evidence
for the live session's preparation and recovery.
