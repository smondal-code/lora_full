"""Experiment runner: executes all solver runs in parallel and writes one JSON per run."""
import glob, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
PSO = os.path.join(ROOT, 'pso')
OUT = os.path.join(ROOT, 'results', 'runs')
os.makedirs(OUT, exist_ok=True)

PROPOSED = ('geo', 'fcr', 1, 'raw')
ABLATION = [('cos', 'fcr', 1, 'raw'), ('geo', 'full', 1, 'raw')]
SUBSET = [f'{p}{i}_21' for p in ('c10', 'c20', 'r10', 'r20') for i in range(1, 5)] + \
         [f'rc{p}{i}_21' for p in ('10', '20') for i in range(1, 5)]


LAMARCK = ('geo', 'fcr', 1, 'lamarck')


def jobs():
    js = []
    # final configuration (Lamarckian credit assignment): 56 large instances x 5 seeds
    for f in sorted(glob.glob(f'{ROOT}/instances/100/*.txt')):
        for seed in range(1, 6):
            js.append((f, *LAMARCK, seed, 30, 150))
    return js


def run(job):
    f, dec, ls, re, guide, seed, swarm, iters = job
    name = os.path.basename(f)[:-4]
    out = f'{OUT}/{name}__{dec}_{ls}_re{re}_{guide}__s{seed}.json'
    if os.path.exists(out):
        return
    subprocess.run([PSO, f, '--decoder', dec, '--ls', ls, '--re', str(re), '--guide', guide, '--seed', str(seed),
                    '--swarm', str(swarm), '--iters', str(iters), '--out', out + '.tmp'],
                   stderr=subprocess.DEVNULL, check=True)
    os.replace(out + '.tmp', out)


if __name__ == '__main__':
    J = jobs()
    print(len(J), 'jobs', flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=2) as ex:
        for _ in ex.map(run, J):
            done += 1
            if done % 50 == 0:
                print(done, flush=True)
    print('ALL DONE', flush=True)
