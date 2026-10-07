"""
Design study for the book chapter: gateway density and spreading-factor (SF)
allocation in a 20 km x 20 km LoRaWAN sensing deployment.

Model (all assumptions are stated in the chapter):
- N nodes uniform in a square of side L; g x g gateways at the centres of an
  equal-area grid; each node attaches to the gateway with the lowest path loss.
- Log-distance path loss with log-normal shadowing:
      PL(d) = PL1 + 10 n log10(d / 1 km) + X,  X ~ N(0, sigma^2)
- 125 kHz, CR 4/5, 23-byte PHY payload, 3 uplink channels, one uplink per T s.
- Sensitivity from S = -174 + 10 log10(BW) + NF + SNR_req (NF = 6 dB).
- Uplink success = exp(-2G) per (gateway, SF) group (pure ALOHA, same-SF
  collisions only, no capture, no inter-cell interference).
- Energy: 29 mA transmit, 10.8 mA receive, 3 uA sleep, two receive windows of
  max(20 ms, 6 symbol times) each, 2400 mAh battery, self-discharge ignored.
Run:  python3 lora_design_study.py
"""
import math, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BW = 125e3
SNR_REQ = {7: -7.5, 8: -10, 9: -12.5, 10: -15, 11: -17.5, 12: -20}
SENS = {sf: -174 + 10*math.log10(BW) + 6 + SNR_REQ[sf] for sf in SNR_REQ}
TX_DBM = 14.0

def airtime(pl, sf, cr=1, crc=1, ih=0, npre=8):
    ts = 2**sf / BW
    de = 1 if ts > 0.016 else 0
    n = 8 + max(math.ceil((8*pl - 4*sf + 28 + 16*crc - 20*ih) / (4*(sf - 2*de))) * (cr + 4), 0)
    return (npre + 4.25) * ts + n * ts

TA = {sf: airtime(23, sf) for sf in SENS}
RXT = {sf: 2 * max(0.02, 6 * 2**sf / BW) for sf in SENS}

def avg_current_ma(sf, interval_s, itx=29.0, irx=10.8, isleep=0.003):
    q = itx * TA[sf] + irx * RXT[sf] + isleep * interval_s
    return q / interval_s

SCEN = {
    "open rural":  dict(PL1=110.0, n=2.7, sigma=6.0),
    "obstructed":  dict(PL1=120.0, n=3.8, sigma=8.0),
}

def run(scen, g, strategy, N=2000, L=20.0, interval=900.0, nch=3,
        adr_margin=10.0, runs=100, seed=1, cap_years=15.0):
    p = SCEN[scen]
    rng = np.random.default_rng(seed)
    cell = L / g
    gx = (np.arange(g) + 0.5) * cell
    gws = np.array([(x, y) for x in gx for y in gx])
    res = dict(cov=[], meansf=[], pdr=[], p5=[], cur=[], life10=[])
    for _ in range(runs):
        xy = rng.uniform(0, L, size=(N, 2))
        d = np.linalg.norm(xy[:, None, :] - gws[None, :, :], axis=2)
        d = np.maximum(d, 0.05)
        pl = p["PL1"] + 10 * p["n"] * np.log10(d) + rng.normal(0, p["sigma"], size=d.shape)
        best = pl.argmin(axis=1)
        plb = pl[np.arange(N), best]
        rx = TX_DBM - plb
        covered = rx >= SENS[12]
        sf = np.full(N, 12)
        if strategy == "adr":
            for s in (12, 11, 10, 9, 8, 7):
                ok = rx >= SENS[s] + adr_margin
                sf[ok] = s
        # collision groups
        pdr = np.zeros(N)
        for gi in range(len(gws)):
            for s in range(7, 13):
                m = (best == gi) & (sf == s) & covered
                cnt = m.sum()
                if cnt:
                    G = cnt * TA[s] / interval / nch
                    pdr[m] = math.exp(-2 * G)
        cur = np.array([avg_current_ma(int(s), interval) for s in sf])
        life = np.minimum(2400.0 / cur / 8760.0, cap_years)
        res["cov"].append(covered.mean())
        res["meansf"].append(sf[covered].mean() if covered.any() else float("nan"))
        res["pdr"].append(pdr[covered].mean() if covered.any() else 0)
        res["p5"].append(np.percentile(pdr[covered], 5) if covered.any() else 0)
        res["cur"].append(np.median(cur[covered]) * 1000 if covered.any() else float("nan"))
        res["life10"].append(((2400.0 / cur / 8760.0) >= 10)[covered].mean() if covered.any() else 0)
    return {k: float(np.nanmean(v)) for k, v in res.items()}


# --------------------------------------------------------------------------
# Graphical results, drawn straight from the `out` dict dumped to JSON above
# (dataviz skill: validated categorical palette, 2px lines, >=8px markers,
# white-filled dashed markers for the SF12-only series, legend, no dual axis).
# --------------------------------------------------------------------------
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"
COL = {"Open rural": BLUE, "Obstructed": ORANGE}
SCEN_KEY = {"Open rural": "open rural", "Obstructed": "obstructed"}
GW = [1, 4, 9, 16, 25]
X_GW = "Number of gateways over 400 km$^2$"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8a86", "axes.linewidth": 0.8,
})


def _base(xlabel, ylabel, ylim=None):
    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=200)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#dddddd", lw=0.7)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)
    return fig, ax


def _finish(fig, name, outdir):
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}.png"))
    plt.close(fig)


def make_figures(out, outdir="figs"):
    """Build coverage, delivery, current, battery-life and density charts
    from the simulation results dict (same dict that gets json.dump'ed)."""
    os.makedirs(outdir, exist_ok=True)

    # Coverage (strategy-independent; read off the sf12 runs)
    fig, ax = _base(X_GW, "Coverage (%)", (0, 105))
    for s in COL:
        y = [out[f"{SCEN_KEY[s]}|{g}|sf12"]["cov"] * 100 for g in GW]
        ax.plot(GW, y, "-o", color=COL[s], lw=2, ms=8, mfc=COL[s], mec="white", mew=1, label=s)
    ax.set_xticks(GW)
    ax.legend(frameon=False, loc="lower right")
    _finish(fig, "coverage", outdir)

    # Delivery ratio: ADR-like (solid) vs SF12-only (dashed)
    fig, ax = _base(X_GW, "Delivery ratio (%)", (0, 105))
    for s in COL:
        y_adr = [out[f"{SCEN_KEY[s]}|{g}|adr"]["pdr"] * 100 for g in GW]
        y_sf12 = [out[f"{SCEN_KEY[s]}|{g}|sf12"]["pdr"] * 100 for g in GW]
        ax.plot(GW, y_adr, "-o", color=COL[s], lw=2, ms=8, mfc=COL[s], mec="white", mew=1,
                label=f"{s}, ADR-like")
        ax.plot(GW, y_sf12, "--o", color=COL[s], lw=2, ms=8, mfc="white", mec=COL[s], mew=1.3,
                label=f"{s}, SF12 only")
    ax.set_xticks(GW)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    _finish(fig, "delivery", outdir)

    # Median current (ADR-like allocation); SF12-only reference line
    fig, ax = _base(X_GW, "Median current (µA)", (0, 62))
    for s in COL:
        y = [out[f"{SCEN_KEY[s]}|{g}|adr"]["cur"] for g in GW]
        ax.plot(GW, y, "-o", color=COL[s], lw=2, ms=8, mfc=COL[s], mec="white", mew=1,
                label=f"{s}, ADR-like")
    sf12_cur = out[f"{SCEN_KEY['Open rural']}|1|sf12"]["cur"]
    ax.axhline(sf12_cur, color=GREY, ls="--", lw=1)
    ax.text(25, sf12_cur + 2, f"SF12 only: {sf12_cur:.1f} µA", ha="right", color=GREY, fontsize=8)
    ax.set_xticks(GW)
    ax.legend(frameon=False, loc="center right")
    _finish(fig, "current", outdir)

    # Share of covered nodes with ideal battery life >= 10 years (ADR-like)
    fig, ax = _base(X_GW, "Nodes with ideal life ≥ 10 years (%)", (0, 105))
    for s in COL:
        y = [out[f"{SCEN_KEY[s]}|{g}|adr"]["life10"] * 100 for g in GW]
        ax.plot(GW, y, "-o", color=COL[s], lw=2, ms=8, mfc=COL[s], mec="white", mew=1, label=s)
    ax.set_xticks(GW)
    ax.legend(frameon=False, loc="lower right")
    _finish(fig, "life", outdir)

    # Delivery ratio vs node density (9 gateways)
    n_labels, n_vals = ["500", "2,000", "8,000"], [500, 2000, 8000]
    fig, ax = _base("Nodes over 400 km$^2$ (9 gateways)", "Delivery ratio (%)", (0, 105))
    for s in COL:
        y_adr = [out[f"dens|{SCEN_KEY[s]}|{n}|adr"]["pdr"] * 100 for n in n_vals]
        y_sf12 = [out[f"dens|{SCEN_KEY[s]}|{n}|sf12"]["pdr"] * 100 for n in n_vals]
        ax.plot(n_labels, y_adr, "-o", color=COL[s], lw=2, ms=8, mfc=COL[s], mec="white", mew=1,
                label=f"{s}, ADR-like")
        ax.plot(n_labels, y_sf12, "--o", color=COL[s], lw=2, ms=8, mfc="white", mec=COL[s], mew=1.3,
                label=f"{s}, SF12 only")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    _finish(fig, "density", outdir)

    print(f"figures written to {outdir}/")


if __name__ == "__main__":
    out = {}
    print("Airtime (ms):", {s: round(TA[s]*1000, 1) for s in TA})
    print("Sensitivity (dBm):", {s: round(SENS[s], 1) for s in SENS})
    for scen in SCEN:
        for g in (1, 2, 3, 4, 5):
            for strat in ("sf12", "adr"):
                r = run(scen, g, strat)
                out[f"{scen}|{g*g}|{strat}"] = r
                print(scen, g*g, strat, {k: round(v, 3) for k, v in r.items()})
    # sensitivity to reporting interval and channel count (9 gateways, ADR)
    for scen in SCEN:
        for interval in (300.0, 900.0, 3600.0):
            for nch in (3, 8):
                r = run(scen, 3, "adr", interval=interval, nch=nch)
                out[f"sens|{scen}|{int(interval)}|{nch}"] = r
                print("sens", scen, int(interval), nch, {k: round(v, 3) for k, v in r.items()})
    # density sensitivity: more nodes, 9 gateways, ADR vs SF12
    for scen in SCEN:
        for N in (500, 2000, 8000):
            for strat in ("sf12", "adr"):
                r = run(scen, 3, strat, N=N)
                out[f"dens|{scen}|{N}|{strat}"] = r
                print("dens", scen, N, strat, {k: round(v, 3) for k, v in r.items()})
    json.dump(out, open("lora_design_study_results.json", "w"), indent=1)
    make_figures(out, outdir="figs")
