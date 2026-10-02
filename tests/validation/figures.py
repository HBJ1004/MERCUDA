"""Final scientific figures; no raw trajectories or timing benchmark claims."""
import json
import hashlib
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(output):
    report = json.loads((output/'report.json').read_text())
    grouped = defaultdict(lambda:[0,0,0])
    for row in report['cases']:
        grouped[row['group']][{'pass':0,'fail':1,'incomplete':2}[row['status']]] += 1
    names = sorted(grouped)
    fig,ax = plt.subplots(figsize=(10,7),constrained_layout=True)
    colors = ['#174f82','#bd3f3f','#d09b39']
    for index,label in enumerate(['Passed','Failed','Incomplete']):
        values = [grouped[name][index] for name in names]
        rows = [i for i,value in enumerate(values) if value]
        ax.barh([i+(index-1)*.23 for i in rows], [values[i] for i in rows],
                height=.22,color=colors[index],label=label)
        for i in rows:
            ax.text(values[i]*1.08,i+(index-1)*.23,format(values[i],','),
                    va='center',fontsize=8,color=colors[index])
    ax.set_yticks(range(len(names)),[name.replace('-',' ').capitalize() for name in names])
    ax.invert_yaxis(); ax.set_xscale('log'); ax.set_xlim(.7,max(sum(grouped[n]) for n in names)*3)
    ax.set_xlabel('Validation cases (logarithmic scale)')
    ax.set_title('Validation coverage · '+report['status'].capitalize()); ax.legend(loc='lower right')
    ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True)
    fig.savefig(output/'coverage.png',dpi=180); plt.close(fig)
    curves = defaultdict(list)
    for row in report['cases']:
        if row['status'] != 'pass': continue
        for entry in row.get('metrics',{}).get('convergence',[]):
            curves[row['settings'].get('method','unknown')].append(entry['errors'])
    if curves:
        fig,ax = plt.subplots(figsize=(7,4.5),constrained_layout=True)
        for method,series in sorted(curves.items()):
            maximum = [max(values[i] for values in series) for i in range(3)]
            ax.semilogy([0,1,2],maximum,'o-',linewidth=2.5,markersize=7,label=method)
        ax.set_xticks([0,1,2],['Coarse','Intermediate','Fine'])
        ax.set_ylabel('Maximum normalized state error'); ax.set_xlabel('Refinement level')
        ax.grid(alpha=.2); ax.legend(loc='upper right'); ax.set_title('Trajectory convergence')
        fig.savefig(output/'convergence.png',dpi=180); plt.close(fig)
    artifacts = {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                 for name in ('coverage.png','convergence.png') if (output/name).exists()}
    provenance = {'rendered_utc':datetime.now(timezone.utc).isoformat(),
                  'report_sha256':hashlib.sha256((output/'report.json').read_bytes()).hexdigest(),
                  'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'artifacts_sha256':artifacts}
    (output/'rendering.json').write_text(json.dumps(provenance,indent=2)+'\n')
