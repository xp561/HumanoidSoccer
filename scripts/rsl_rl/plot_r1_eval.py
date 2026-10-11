import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

out=Path('logs/rsl_rl/r1_flat/2026-10-02_02-40-03_r1_first/evaluation/r1_robustness')
d=json.loads((out/'summary.json').read_text())
rows=d['trials']
pos=defaultdict(list); roll=defaultdict(list)
for r in rows:
 if r['group']=='position_grid': pos[r['condition']].append(r)
 if r['group']=='rolling_grid': roll[r['condition']].append(r)
for metric,title,name in [('success_goal_rate','Success rate','position_success.png'),('kick_accuracy_cosine','Cosine accuracy','position_accuracy.png')]:
 mat=np.zeros((3,3)); fw=[.6,.9,1.2]; lat=[-.3,0,.3]
 for i,f in enumerate(fw):
  for j,l in enumerate(lat): mat[i,j]=np.mean([x[metric] for x in pos[f'forward={f:.1f},lateral={l:.1f}']])
 fig,ax=plt.subplots();im=ax.imshow(mat,vmin=0 if metric=='success_goal_rate' else .95,vmax=1,cmap='viridis');ax.set(xticks=range(3),xticklabels=lat,yticks=range(3),yticklabels=fw,xlabel='lateral offset (m)',ylabel='forward distance (m)',title=title);fig.colorbar(im,ax=ax);fig.savefig(out/name,dpi=160,bbox_inches='tight');plt.close(fig)
keys=sorted(roll); values=[np.mean([x['success_goal_rate'] for x in roll[k]]) for k in keys]; acc=[np.mean([x['kick_accuracy_cosine'] for x in roll[k]]) for k in keys]
fig,ax=plt.subplots();x=np.arange(len(keys));ax.bar(x-.18,values,.36,label='success rate');ax.bar(x+.18,acc,.36,label='cosine accuracy');ax.set(xticks=x,xticklabels=keys,ylabel='score',ylim=(0,1.05),title='Rolling-ball robustness');ax.legend();fig.autofmt_xdate(rotation=35);fig.savefig(out/'rolling_robustness.png',dpi=160,bbox_inches='tight')
