# R1 Evaluation Changelog

## 2026-10-11

- Added deterministic, vectorized Isaac Lab soccer evaluation controls for fixed ball placement, rolling-ball speed sampling, initial robot displacement, and selected robustness perturbations.
- Reused the original MuJoCo `goal_crossed` predicate for R1 success measurement and recorded outgoing-ball direction as a cosine metric.
- Added paper-aligned static and rolling evaluation aggregation for the 11 x 13 ball-position grid, including Table IV workspace filtering, Wilson confidence intervals, per-position CSV output, and heatmaps.
- Added a separate R1 Standard-motion tracking evaluator. Its global/local position and root finite-difference velocity metrics follow the repository's existing definitions. The peak-link-jerk statistic is explicitly reported as an AUJ proxy because the exact paper AUJ aggregation is not implemented in the original repository.
- Generated the current results from `model_103998.pt` using the ten converted Standard R1 motion files. Stylized-motion comparison remains unavailable until matching R1 motions are supplied.
