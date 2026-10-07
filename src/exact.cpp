// Exact solver for small E-VRPTW instances (<= ~15 customers), full-recharge model.
// Phase 1: forward label setting enumerates, for every customer subset T, the minimum-distance
//          feasible elementary route serving exactly T (unlimited station visits allowed).
// Phase 2: set-partitioning dynamic programme over subsets with lexicographic objective
//          (number of vehicles, then total distance). Also reports the distance-only optimum.
#include "common.h"
#include <chrono>
#include <unordered_map>

struct XL { double dist, t, b; };

int main(int argc, char** argv) {
    Instance I;
    I.load(argv[1]);
    auto t0 = std::chrono::steady_clock::now();
    int n = I.customers.size();
    int full = 1 << n;
    std::vector<double> routeCost(full, INF);
    // node indexing: 0 depot, stations, customers (as in Instance)
    auto key = [&](int set, int node) { return (long long)set * I.n + node; };
    std::vector<std::unordered_map<long long, std::vector<XL>>> level(n + 1);
    level[0][key(0, 0)].push_back({0, 0, I.Q});
    const double depotDue = I.nodes[0].due;
    double minToCharge = INF;  // not used for pruning beyond direct checks
    auto dominatedInsert = [&](std::vector<XL>& vec, const XL& nl) {
        for (auto& o : vec)
            if (o.dist <= nl.dist + EPS && o.t <= nl.t + EPS && o.b >= nl.b - EPS) return false;
        vec.erase(std::remove_if(vec.begin(), vec.end(), [&](const XL& o) {
                      return nl.dist <= o.dist + EPS && nl.t <= o.t + EPS && nl.b >= o.b - EPS; }), vec.end());
        vec.push_back(nl);
        return true;
    };
    // minimal distance from each node to any station or depot (energy reachability pruning)
    std::vector<double> dEscape(I.n, INF);
    for (int a = 0; a < I.n; a++) {
        dEscape[a] = I.d[a][0];
        for (int s : I.stations) if (s != a) dEscape[a] = std::min(dEscape[a], I.d[a][s]);
    }
    long long labels = 0;
    for (int k = 0; k <= n; k++) {
        // worklist: process labels of this level; station moves stay in this level
        std::vector<std::pair<long long, XL>> work;
        for (auto& kv : level[k]) for (auto& l : kv.second) work.push_back({kv.first, l});
        size_t wi = 0;
        while (wi < work.size()) {
            auto [kk, l] = work[wi++];
            int set = kk / I.n, node = kk % I.n;
            // label may have been dominated after being queued
            auto& curv = level[k][kk];
            bool alive = false;
            for (auto& o : curv) if (std::fabs(o.dist - l.dist) < 1e-9 && std::fabs(o.t - l.t) < 1e-9 && std::fabs(o.b - l.b) < 1e-9) alive = true;
            if (!alive) continue;
            labels++;
            // close route at depot
            if (set != 0 && node != 0) {
                double dd = I.d[node][0];
                if (l.b - I.r * dd >= -EPS && l.t + dd / I.v <= depotDue + EPS)
                    routeCost[set] = std::min(routeCost[set], l.dist + dd);
            }
            double load = 0;
            for (int i = 0; i < n; i++) if (set >> i & 1) load += I.nodes[I.customers[i]].demand;
            // to customers
            for (int i = 0; i < n; i++) {
                if (set >> i & 1) continue;
                int c = I.customers[i];
                const Node& Cn = I.nodes[c];
                if (load + Cn.demand > I.C + EPS) continue;
                double dd = I.d[node][c];
                double b = l.b - I.r * dd;
                if (b < -EPS) continue;
                double t = l.t + dd / I.v;
                if (t > Cn.due + EPS) continue;
                t = std::max(t, Cn.ready) + Cn.service;
                if (t + I.d[c][0] / I.v > depotDue + EPS) continue;
                if (b - I.r * dEscape[c] < -EPS) continue;
                int ns = set | (1 << i);
                dominatedInsert(level[k + 1][key(ns, c)], {l.dist + dd, t, b});
            }
            // to stations (full recharge)
            for (int s : I.stations) {
                if (s == node) continue;
                double dd = I.d[node][s];
                if (dd < EPS) continue;
                double b = l.b - I.r * dd;
                if (b < -EPS) continue;
                double t = l.t + dd / I.v;
                if (t > I.nodes[s].due + EPS) continue;
                t += I.g * (I.Q - std::max(0.0, b));
                if (t + I.d[s][0] / I.v > depotDue + EPS) continue;
                XL nl{l.dist + dd, t, I.Q};
                if (dominatedInsert(level[k][key(set, s)], nl)) work.push_back({key(set, s), nl});
            }
        }
        level[k].clear();
    }
    // Phase 2: partitioning DP
    std::vector<int> bnv(full, 1 << 29);
    std::vector<double> bd(full, INF), dOnly(full, INF);
    std::vector<int> choice(full, -1);
    bnv[0] = 0; bd[0] = 0; dOnly[0] = 0;
    for (int S = 1; S < full; S++) {
        int low = S & -S;
        int rest = S ^ low;
        // enumerate subsets T of S containing low
        for (int sub = rest;; sub = (sub - 1) & rest) {
            int T = sub | low;
            if (routeCost[T] < INF) {
                int R = S ^ T;
                if (bnv[R] < (1 << 29)) {
                    int nv = bnv[R] + 1;
                    double dist = bd[R] + routeCost[T];
                    if (nv < bnv[S] || (nv == bnv[S] && dist < bd[S] - 1e-9)) { bnv[S] = nv; bd[S] = dist; choice[S] = T; }
                }
                if (dOnly[R] < INF) dOnly[S] = std::min(dOnly[S], dOnly[R] + routeCost[T]);
            }
            if (sub == 0) break;
        }
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    int S = full - 1;
    printf("{\"instance\":\"%s\",\"n\":%d,\"nv\":%d,\"td\":%.4f,\"td_distance_only\":%.4f,\"seconds\":%.3f,\"labels\":%lld,\"routes\":[",
           I.name.c_str(), n, bnv[S], bd[S], dOnly[S], secs, labels);
    bool first = true;
    while (S) {
        int T = choice[S];
        printf("%s[", first ? "" : ",");
        first = false;
        bool f2 = true;
        for (int i = 0; i < n; i++) if (T >> i & 1) { printf("%s\"%s\"", f2 ? "" : ",", I.nodes[I.customers[i]].id.c_str()); f2 = false; }
        printf("]");
        S ^= T;
    }
    printf("]}\n");
    return 0;
}
