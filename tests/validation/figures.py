"""Final scientific figures; no raw trajectories or timing benchmark claims."""
import json
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(output):
    report = json.loads((output/'report.json').read_text())
    grouped = defaultdict(lambda:[0,0,0])
    for row in report['cases']:
        grouped[row['group']][{'pass':0,'fail':1,'incomplete':2}[row['status']]] += 1
    names = sorted(grouped)
    fig,ax = plt.subplots(figsize=(10,4.5),constrained_layout=True)
    passed = [grouped[name][0] for name in names]; failed = [grouped[name][1] for name in names]
    ax.bar(names,passed,color='#174f82',label='Passed')
    ax.bar(names,failed,bottom=passed,color='#bd3f3f',label='Failed')
    incomplete = [grouped[name][2] for name in names]
    ax.bar(names,incomplete,bottom=[a+b for a,b in zip(passed,failed)],color='#d09b39',label='Incomplete')
    ax.set_ylabel('Validation cases'); ax.tick_params(axis='x',rotation=55)
    ax.set_title('Validation coverage — '+report['status']); ax.legend()
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
        ax.grid(alpha=.2); ax.legend(); ax.set_title('Trajectory convergence')
        fig.savefig(output/'convergence.png',dpi=180); plt.close(fig)
