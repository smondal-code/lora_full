"""Aggregate all runs into the tables used in the paper (results/summary.json + printed tables)."""
import json, glob, os, collections, statistics as st
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.abspath(__file__))
runs = collections.defaultdict(list)
for f in glob.glob(f'{ROOT}/results/runs/*.json'):
    d = json.loads(open(f).readline())
    runs[(d['instance'], f"{d['decoder']}_{d['ls']}_re{d['re']}" + ('_raw' if d.get('guide') == 'raw' else ''))].append(d)

exact = {}
for f in glob.glob(f'{ROOT}/results/exact/*.jsonl'):
    for l in open(f):
        d = json.loads(l)
        exact[d['instance']] = d

PROP, BASE, IMPR = 'geo_fcr_re1_raw', 'cos_none_re0', 'geo_fcr_re1'
VARIANTS = ['cos_none_re0', 'cos_fcr_re1_raw', 'geo_none_re0', 'geo_fcr_re1', 'geo_fcr_re1_raw', 'geo_full_re1_raw', 'cos_fcr_re1', 'geo_full_re1']
NAMES = {'cos_none_re0': 'PSODS-II-EV', 'cos_fcr_re1_raw': 'PSODS-II-EV + LS', 'geo_none_re0': 'Geo decoder only',
         'geo_fcr_re1': 'E-PSODS improved-guidance', 'geo_fcr_re1_raw': 'E-PSODS', 'geo_full_re1_raw': 'E-PSODS full LS',
         'cos_fcr_re1': 'PSODS-II-EV + LS (improved-guidance)', 'geo_full_re1': 'E-PSODS full LS (improved-guidance)'}


def cls(name):
    base = name.split('_')[0] if name.endswith('_21') else name.split('C')[0]
    t = ''.join(ch for ch in base if ch.isalpha())
    return t.upper() + base[len(t)]


def stats(rs, seeds=None):
    if seeds:
        rs = [r for r in rs if r['seed'] in seeds]
    best = min(rs, key=lambda r: (r['nv'], r['td']))
    return dict(n=len(rs), best_nv=best['nv'], best_td=best['td'],
                mean_nv=st.mean(r['nv'] for r in rs), mean_td=st.mean(r['td'] for r in rs),
                sd_td=st.pstdev(r['td'] for r in rs), time=st.mean(r['seconds'] for r in rs),
                fit=[r['nv'] * 1e5 + r['td'] for r in rs])


out = {}
# ---------- small instances ----------
small = []
for inst in sorted(exact, key=lambda s: (int(s.split('C')[-1]), s)):
    e = exact[inst]
    row = dict(instance=inst, size=int(inst.split('C')[-1]), cls=cls(inst), ex_nv=e['nv'], ex_td=e['td'],
               ex_time=e['seconds'])
    for v in (PROP, BASE, IMPR):
        rs = runs.get((inst, v), [])
        if not rs:
            continue
        s = stats(rs)
        hits = sum(1 for r in rs if r['nv'] == e['nv'] and abs(r['td'] - e['td']) < 0.01)
        gap = [(r['td'] - e['td']) / e['td'] * 100 for r in rs if r['nv'] == e['nv']]
        row[v] = dict(best_nv=s['best_nv'], best_td=s['best_td'], mean_td=s['mean_td'], mean_nv=s['mean_nv'],
                      hits=hits, n=s['n'], time=s['time'], gap_best=(s['best_td'] - e['td']) / e['td'] * 100
                      if s['best_nv'] == e['nv'] else None,
                      extra_nv_runs=sum(1 for r in rs if r['nv'] > e['nv']),
                      mean_gap_same_nv=st.mean(gap) if gap else None)
    small.append(row)
out['small'] = small

# ---------- large instances, proposed ----------
large = []
for inst in sorted({k[0] for k in runs if k[0].endswith('_21')}):
    rs = runs.get((inst, PROP), [])
    if not rs:
        continue
    s = stats(rs)
    s.pop('fit')
    # charging statistics of the best solution
    best = min(rs, key=lambda r: (r['nv'], r['td']))
    visits = sum(1 for r in best['routes'] for x in r[1:-1] if x.startswith('S'))
    doubles = sum(1 for r in best['routes'] for a, b in zip(r, r[1:]) if a.startswith('S') and b.startswith('S'))
    routes_w = sum(1 for r in best['routes'] if any(x.startswith('S') for x in r[1:-1]))
    ri = runs.get((inst, IMPR), [])
    si = stats(ri) if ri else None
    if si: si.pop('fit')
    large.append(dict(instance=inst, cls=cls(inst), **s, station_visits=visits, double_stops=doubles,
                      routes_with_charging=routes_w, K=best['K'], improved=si))
out['large'] = large

# ---------- ablation ----------
subset = sorted({k[0] for k in runs if k[1] == 'cos_none_re0' and k[0].endswith('_21')})
abl = {}
for v in VARIANTS:
    rows = []
    for inst in subset:
        rs = runs.get((inst, v), [])
        if not rs:
            continue
        s = stats(rs, seeds=set(range(1, 4)))
        rows.append(dict(instance=inst, cls=cls(inst), **{k: s[k] for k in s if k != 'fit'}))
    abl[v] = rows
out['ablation'] = abl


def compare(a, b, key_nv='mean_nv', key_td='mean_td'):
    """Paired comparison of variant a vs b over common instances."""
    A = {r['instance']: r for r in abl[a]}
    B = {r['instance']: r for r in abl[b]};
    common = sorted(set(A) & set(B))
    nv_better = sum(1 for i in common if A[i][key_nv] < B[i][key_nv] - 1e-9)
    nv_worse = sum(1 for i in common if A[i][key_nv] > B[i][key_nv] + 1e-9)
    fa = [A[i][key_nv] * 1e5 + A[i][key_td] for i in common]
    fb = [B[i][key_nv] * 1e5 + B[i][key_td] for i in common]
    p = wilcoxon(fa, fb).pvalue if any(x != y for x, y in zip(fa, fb)) else 1.0
    same = [i for i in common if abs(A[i][key_nv] - B[i][key_nv]) < 1e-9]
    td_better = sum(1 for i in same if A[i][key_td] < B[i][key_td])
    ptd = wilcoxon([A[i][key_td] for i in same], [B[i][key_td] for i in same]).pvalue if len(same) >= 6 else None
    return dict(n=len(common), nv_better=nv_better, nv_worse=nv_worse, p_lex=float(p), same_nv=len(same),
                td_better_same_nv=td_better, p_td_same_nv=None if ptd is None else float(ptd))


out['tests'] = {v: compare(PROP, v) for v in VARIANTS if v != PROP and abl.get(v)}
# small instances: proposed vs baseline (mean lexicographic fitness per instance)
def small_test(A, B):
    fa = [r[A]['mean_nv'] * 1e5 + r[A]['mean_td'] for r in small if A in r and B in r]
    fb = [r[B]['mean_nv'] * 1e5 + r[B]['mean_td'] for r in small if A in r and B in r]
    return dict(n=len(fa), better=int(sum(a < b - 1e-9 for a, b in zip(fa, fb))),
                worse=int(sum(a > b + 1e-9 for a, b in zip(fa, fb))), p=float(wilcoxon(fa, fb).pvalue))


out['tests_small'] = {'vs_baseline': small_test(PROP, BASE), 'vs_improved': small_test(PROP, IMPR),
                      'improved_vs_baseline': small_test(IMPR, BASE)}
json.dump(out, open(f'{ROOT}/results/summary.json', 'w'), indent=1)

# ---------- print ----------
print('SMALL')
for size in (5, 10, 15):
    rows = [r for r in small if r['size'] == size and PROP in r]
    for v in (PROP, BASE, IMPR):
        opt = sum(1 for r in rows if r[v]['best_nv'] == r['ex_nv'] and abs(r[v]['best_td'] - r['ex_td']) < 0.01)
        hits = sum(r[v]['hits'] for r in rows)
        n = sum(r[v]['n'] for r in rows)
        extra = sum(r[v]['extra_nv_runs'] for r in rows)
        g = [r[v]['mean_gap_same_nv'] for r in rows if r[v]['mean_gap_same_nv'] is not None]
        print(size, NAMES[v], f'opt(best)={opt}/{len(rows)} hit-rate={hits}/{n} extraNVruns={extra} '
              f'meangap={st.mean(g):.2f}% time={st.mean(r[v]["time"] for r in rows):.2f}s')
print('small test', out['tests_small'])
print('LARGE by class')
by = collections.defaultdict(list)
for r in large:
    by[r['cls']].append(r)
for c in sorted(by):
    rs = by[c]
    print(c, len(rs), 'bestNV %.2f bestTD %.2f meanNV %.2f meanTD %.2f sd %.2f time %.1f visits %.1f doubles %.2f' % (
        st.mean(r['best_nv'] for r in rs), st.mean(r['best_td'] for r in rs), st.mean(r['mean_nv'] for r in rs),
        st.mean(r['mean_td'] for r in rs), st.mean(r['sd_td'] for r in rs), st.mean(r['time'] for r in rs),
        st.mean(r['station_visits'] for r in rs), st.mean(r['double_stops'] for r in rs)))
print('ABLATION')
for v in VARIANTS:
    rs = abl.get(v, [])
    if rs:
        print(NAMES[v], len(rs), 'meanNV %.3f meanTD %.2f bestNV %.3f time %.1f' % (
            st.mean(r['mean_nv'] for r in rs), st.mean(r['mean_td'] for r in rs),
            st.mean(r['best_nv'] for r in rs), st.mean(r['time'] for r in rs)))
for k, t in out['tests'].items():
    print('E-PSODS vs', NAMES[k], t)
