# R1 Training Pipeline Change Log

## 2026-10-01 (Europe/Berlin)

- Added the R1 flat, terrain, and soccer task configurations, with the two head
  joints excluded from policy actions.
- Added R1 actuator setup, joint-name adaptations, and the R1 URDF mesh assets
  required by Isaac Sim.
- Added conversion of the G1 standard motion set into R1 kinematic archives and
  recorded joint/body ordering metadata in those archives.
- Added asset-load and environment smoke-test utilities, including a test of
  the R1 terrain training stage and the SoccerMoving task.
- Corrected startup joint-offset randomization so articulation indices remain
  aligned with the smaller R1 policy action vector.
- Added the two-stage R1 curriculum launcher.  It follows the G1 sequence and
  uses only the converted standard motion set in both stages.

## Validation

- R1 loaded in Isaac Sim with 26 joints, 27 bodies, and 24 policy action joints.
- All converted standard motions passed structural and finite-value validation.
- The R1 flat and SoccerMoving environments completed reset-and-step smoke tests.
- The R1 terrain RNN training stage completed a short 64-environment, two-iteration run.
