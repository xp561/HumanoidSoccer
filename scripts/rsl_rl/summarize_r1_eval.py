from __future__ import annotations
import csv, json
from pathlib import Path
from collections import defaultdict

root = Path('/tmp')
groups = {
 'ball_forward': [('0.6', root/'r1_eval_forward_0.6.json'), ('0.9', root/'r1_eval_forward_0.9.json'), ('1.2', root/'r1_eval_forward_1.2.json')],
 'ball_lateral': [('-0.3', root/'r1_eval_lateral_-0.3.json'), ('0.0', root/'r1_eval_lateral_0.0.json'), ('0.3', root/'r1_eval_lateral_0.3.json')],
 'rolling': [(f'{s}_{d}', root/f'r1_eval_rolling_{s}_{d}.json') for s in ('0.15','0.30') for d in ('0','90','180')],
 'friction': [('0.4',root/'r1_eval_material_0.4.json'),('1.0',root/'r1_eval_material_1.0.json'),('1.4',root/'r1_eval_material_1.4.json')],
 'observation_noise': [('0.00',root/'r1_eval_noise_0.00.json'),('0.05',root/'r1_eval_noise_0.05.json'),('0.10',root/'r1_eval_noise_0.10.json')],
 'initial_yaw_deg': [('-30',root/'r1_eval_yaw_-30.json'),('0',root/'r1_eval_yaw_0.json'),('30',root/'r1_eval_yaw_30.json')],
 'push_speed': [('0.0',root/'r1_eval_push_0.0.json'),('0.5',root/'r1_eval_push_0.5.json'),('1.0',root/'r1_eval_push_1.0.json')],
}
rows=[]
for group, items in groups.items():
 for condition, path in items:
  if not path.exists(): raise FileNotFoundError(path)
  row={'group':group,'condition':condition,**json.loads(path.read_text())}
  rows.append(row)
for path in sorted(root.glob('r1_eval_grid_*.json')):
    _, _, _, forward, lateral, seed = path.stem.split('_')
    rows.append({'group':'position_grid', 'condition':f'forward={forward},lateral={lateral}', 'seed':int(seed), **json.loads(path.read_text())})
for path in sorted(root.glob('r1_eval_rolling_repeat_*.json')):
    _, _, _, _, speed, direction, seed = path.stem.split('_')
    rows.append({'group':'rolling_grid', 'condition':f'speed={speed},direction_deg={direction}', 'seed':int(seed), **json.loads(path.read_text())})
out=Path('logs/rsl_rl/r1_flat/2026-10-02_02-40-03_r1_first/evaluation/r1_robustness')
out.mkdir(parents=True,exist_ok=True)
with (out/'trials.csv').open('w',newline='',encoding='utf-8') as stream:
 fieldnames=sorted({key for row in rows for key in row})
 writer=csv.DictWriter(stream,fieldnames=fieldnames);writer.writeheader();writer.writerows(rows)
grouped=defaultdict(list)
for row in rows:
    grouped[(row['group'], row['condition'])].append(row)
aggregates=[]
for (group, condition), group_rows in grouped.items():
    valid_accuracy=[row['kick_accuracy_cosine'] for row in group_rows if row['kick_accuracy_cosine'] is not None]
    aggregates.append({'group':group, 'condition':condition, 'trials':len(group_rows), 'success_goal_rate':sum(row['success_goal_rate'] for row in group_rows)/len(group_rows), 'first_contact_rate':sum(row['first_contact_rate'] for row in group_rows)/len(group_rows), 'kick_accuracy_cosine':sum(valid_accuracy)/len(valid_accuracy) if valid_accuracy else None})
summary={'checkpoint':'model_103998.pt','motion':'soccer-standard-001_right.npz','trials':rows,'aggregates':aggregates,'notes':['Rows in position_grid and rolling_grid use three independent seeds per condition.','Success uses exp.mujoco_soccer.metrics.goal_crossed.','AUJ is not implemented in this repository.']}
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(out)
