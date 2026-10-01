# Soccer Environment Change Log

## 2026-10-01 (Europe/Berlin)

### Baseline before the Isaac Sim compatibility repair

Environment: `soccer`

| Package | Installed version |
| --- | --- |
| `numpy` | `2.2.6` |
| `scipy` | `1.15.3` |
| `isaacsim` | `4.5.0.0` |
| `isaacsim-core` | `4.5.0.0` |

### Reason for the planned change

Isaac Sim's installed package metadata requires `numpy<2.0.0`.  The current
NumPy 2.x installation produces a binary ABI error while Isaac Sim loads
SciPy-dependent extensions.

### Planned repair

Within the `soccer` environment, install a NumPy 1.x version selected by the
existing package resolver:

```bash
python -m pip install 'numpy<2'
```

### Applied repair

NumPy was changed from `2.2.6` to `1.26.4` in the `soccer` environment.

The package resolver also reported pre-existing constraints that remain outside
this repair's scope:

- `ml-dtypes` requires NumPy 2.x.
- The installed `onnx` version differs from the version required by Isaac Lab.

Neither package was changed.

### Rollback

To restore the pre-change NumPy version in the same environment:

```bash
python -m pip install 'numpy==2.2.6'
```
