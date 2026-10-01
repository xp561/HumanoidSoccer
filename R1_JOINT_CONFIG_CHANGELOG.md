# R1 Joint Configuration Change Log

## 2026-10-01 (Europe/Berlin)

### Change

- Added the R1 actuator armature, stiffness, and damping aliases used by the
  R1 joint-group configuration.
- Mapped each R1 joint group to an existing actuator class already used by the
  project, so the R1 articulation configuration no longer references undefined
  actuator parameters.

### Basis

- Reused the actuator classes and the PD gain derivation already established in
  the project's G1 robot configuration.
- Preserved the R1 joint grouping already present in the R1 configuration and
  did not change URDF effort or velocity limits.

### Validation

- Python syntax compilation passed.
- Static symbol validation confirmed that every R1 actuator parameter reference
  is defined.
- Full Isaac Lab configuration loading remains to be checked in an Isaac
  Sim/Kit runtime. No simulator or training job was started for this change.
