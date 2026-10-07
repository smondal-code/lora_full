"""Independent feasibility checker for E-VRPTW solutions produced by the solver.

Re-simulates every route (with its charging-station visits) from the raw instance file and checks:
all customers served exactly once, load capacity, battery never negative (full recharge at stations),
time windows at customers, stations and depot. Recomputes total distance and vehicle count.
"""
import json, math, sys


def load_instance(path):
    nodes, params = {}, {}
    with open(path) as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '/' in line:
                params[line[0]] = float(line.split('/')[1])
                continue
            p = line.split()
            nodes[p[0]] = dict(type=p[1], x=float(p[2]), y=float(p[3]), demand=float(p[4]),
                               ready=float(p[5]), due=float(p[6]), service=float(p[7]))
    return nodes, params


def check(sol, inst_path, tol=1e-4):
    nodes, P = load_instance(inst_path)
    Q, C, r, g, v = P['Q'], P['C'], P['r'], P['g'], P['v']
    depot = [k for k, n in nodes.items() if n['type'] == 'd'][0]
    customers = {k for k, n in nodes.items() if n['type'] == 'c'}
    seen, total, errors = [], 0.0, []
    for ri, route in enumerate(sol['routes']):
        if route[0] != depot or route[-1] != depot:
            errors.append(f'route {ri} does not start/end at depot')
        t, bat, load = 0.0, Q, 0.0
        for a, b in zip(route, route[1:]):
            A, B = nodes[a], nodes[b]
            d = math.hypot(A['x'] - B['x'], A['y'] - B['y'])
            total += d
            t += d / v
            bat -= r * d
            if bat < -tol:
                errors.append(f'route {ri}: battery negative arriving at {b} ({bat:.3f})')
            if t > B['due'] + tol:
                errors.append(f'route {ri}: late at {b} (t={t:.2f} > {B["due"]})')
            if B['type'] == 'c':
                seen.append(b)
                load += B['demand']
                t = max(t, B['ready']) + B['service']
            elif B['type'] == 'f' and b != route[-1]:
                t += g * (Q - max(bat, 0.0))
                bat = Q
        if load > C + tol:
            errors.append(f'route {ri}: load {load} > {C}')
    if sorted(seen) != sorted(customers):
        missing = customers - set(seen)
        dup = {c for c in seen if seen.count(c) > 1}
        errors.append(f'customer coverage wrong: missing={sorted(missing)[:5]} duplicated={sorted(dup)[:5]}')
    nv = len(sol['routes'])
    return dict(ok=not errors, errors=errors, nv=nv, td=total,
                nv_match=nv == sol['nv'], td_match=abs(total - sol['td']) < 1e-2 * max(1, total) * 1e-2)


if __name__ == '__main__':
    import glob, os
    inst_root = sys.argv[1]
    files = sys.argv[2:]
    bad = 0
    for fp in files:
        for line in open(fp):
            s = json.loads(line)
            size = '100' if s['instance'].endswith('_21') else s['instance'].split('C')[-1]
            res = check(s, os.path.join(inst_root, size, s['instance'] + '.txt'))
            if not (res['ok'] and res['nv_match'] and abs(res['td'] - s['td']) < 0.05):
                bad += 1
                print(fp, s['instance'], s.get('decoder'), s.get('ls'), s['seed'], res['errors'][:3],
                      'reported', s['nv'], s['td'], 'recomputed', res['nv'], round(res['td'], 3))
    print(f'checked {len(files)} files, {bad} problems')
