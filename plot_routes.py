"""Route maps of the best E-PSODS solutions (charging stops marked)."""
import json, glob, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify import load_instance

ROOT = os.path.dirname(os.path.abspath(__file__))


def best_run(inst, variant='geo_fcr_re1_lamarck'):
    rs = [json.loads(open(f).readline()) for f in glob.glob(f'{ROOT}/results/runs/{inst}__{variant}__s*.json')]
    return min(rs, key=lambda r: (r['nv'], r['td']))


def draw(ax, inst):
    size = '100' if inst.endswith('_21') else inst.split('C')[-1]
    nodes, P = load_instance(f'{ROOT}/instances/{size}/{inst}.txt')
    sol = best_run(inst)
    cmap = plt.get_cmap('tab20')
    for k, route in enumerate(sol['routes']):
        xs = [nodes[n]['x'] for n in route]
        ys = [nodes[n]['y'] for n in route]
        ax.plot(xs, ys, '-', lw=1.0, color=cmap(k % 20), alpha=0.9, zorder=2)
    cx = [n['x'] for n in nodes.values() if n['type'] == 'c']
    cy = [n['y'] for n in nodes.values() if n['type'] == 'c']
    ax.scatter(cx, cy, s=9, c='#444444', zorder=3, label='customer')
    used = {x for r in sol['routes'] for x in r[1:-1] if x.startswith('S')}
    for k, n in nodes.items():
        if n['type'] == 'f' and k != 'S0':
            ax.scatter([n['x']], [n['y']], marker='^', s=42 if k in used else 26,
                       facecolor='#d62728' if k in used else 'none', edgecolor='#d62728', lw=1, zorder=4)
    d = [n for n in nodes.values() if n['type'] == 'd'][0]
    ax.scatter([d['x']], [d['y']], marker='s', s=60, c='black', zorder=5)
    visits = sum(1 for r in sol['routes'] for x in r[1:-1] if x.startswith('S'))
    ax.set_title(f"{inst.replace('_21', '')}: NV={sol['nv']}, TD={sol['td']:.1f}, {visits} charging stops", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal', adjustable='datalim')
    for s in ax.spines.values():
        s.set_color('#bbbbbb')


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        out = sys.argv[1]
        insts = sys.argv[2:]
    else:
        out = os.path.join(ROOT, 'route_maps.png')
        insts = ['c101_21', 'r101_21', 'rc101_21', 'c201_21', 'r201_21', 'rc201_21']
    cols = 3
    rows = (len(insts) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.7 * rows))
    for ax, inst in zip(axes.flat, insts):
        draw(ax, inst)
    for ax in list(axes.flat)[len(insts):]:
        ax.axis('off')
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='s', color='black', ls='', label='depot'),
               Line2D([], [], marker='o', color='#444444', ls='', ms=4, label='customer'),
               Line2D([], [], marker='^', color='#d62728', ls='', label='station used'),
               Line2D([], [], marker='^', mfc='none', mec='#d62728', ls='', label='station not used')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=160)
    print('saved', out)
