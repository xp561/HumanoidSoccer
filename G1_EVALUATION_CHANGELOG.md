# G1 Evaluation Changelog

## 2026-10-12

- Evaluated `g1_flat/2026-10-01_01-08-31_test/model_36000.pt` with the same static and rolling ball protocol used for R1.
- Ran the 11 x 13 ball-position grid with 50 parallel trials per position for static and rolling ball conditions.
- Generalized the paper-aligned aggregation utility so G1 and R1 use consistent report labels, confidence intervals, heatmaps, and Table IV workspace filtering.
- Added an automated G1-R1 comparison report using the same goal-crossing predicate and direction-cosine calculation for both checkpoints.
- Preserved tracking and soccer summaries separately to prevent tracking output from overwriting soccer-evaluation metadata.
