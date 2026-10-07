// Common E-VRPTW data structures and route evaluation (full recharge, Schneider et al. 2014 model)
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

static const double EPS = 1e-6;
static const double INF = 1e18;

struct Node {
    std::string id;
    char type;  // d, f, c
    double x, y, demand, ready, due, service;
};

struct Instance {
    std::string name;
    std::vector<Node> nodes;          // 0 = depot, then stations, then customers
    std::vector<int> stations;        // node indices of stations
    std::vector<int> customers;       // node indices of customers
    double Q = 0, C = 0, r = 0, g = 0, v = 1;
    std::vector<std::vector<double>> d;
    // cand[a*n+b] = up to M candidate stations for arc a->b ordered by detour
    std::vector<std::vector<int>> cand;
    int n = 0;
    int M = 3;
    int M2 = 2;  // candidates considered for each station of a two-station detour

    double dist(int a, int b) const { return d[a][b]; }

    void load(const std::string& path) {
        std::ifstream in(path);
        if (!in) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
        name = path.substr(path.find_last_of('/') + 1);
        name = name.substr(0, name.find('.'));
        std::string line;
        std::getline(in, line);  // header
        std::vector<Node> dep, st, cu;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            if (line.find('/') != std::string::npos) {
                double val = atof(line.substr(line.find('/') + 1).c_str());
                if (line[0] == 'Q') Q = val;
                else if (line[0] == 'C') C = val;
                else if (line[0] == 'r') r = val;
                else if (line[0] == 'g') g = val;
                else if (line[0] == 'v') v = val;
                continue;
            }
            std::istringstream ss(line);
            Node nd;
            std::string t;
            if (!(ss >> nd.id >> t >> nd.x >> nd.y >> nd.demand >> nd.ready >> nd.due >> nd.service)) continue;
            nd.type = t[0];
            if (nd.type == 'd') dep.push_back(nd);
            else if (nd.type == 'f') st.push_back(nd);
            else cu.push_back(nd);
        }
        nodes.push_back(dep[0]);
        for (auto& s : st) { stations.push_back(nodes.size()); nodes.push_back(s); }
        for (auto& c : cu) { customers.push_back(nodes.size()); nodes.push_back(c); }
        n = nodes.size();
        d.assign(n, std::vector<double>(n, 0));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                d[i][j] = std::hypot(nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y);
        cand.assign(n * n, {});
        for (int a = 0; a < n; a++)
            for (int b = 0; b < n; b++) {
                std::vector<std::pair<double, int>> v2;
                for (int s : stations) {
                    if (d[a][s] < EPS || d[s][b] < EPS) continue;  // co-located station adds nothing
                    v2.push_back({d[a][s] + d[s][b], s});
                }
                std::sort(v2.begin(), v2.end());
                for (int k = 0; k < (int)v2.size() && k < M; k++) cand[a * n + b].push_back(v2[k].second);
            }
    }
};

// Result of evaluating a customer sequence: best station insertion via Pareto labels.
struct RouteEval {
    bool feasible = false;
    double dist = INF;
    std::vector<int> path;  // full path incl. depot at both ends and stations
};

struct Label {
    double dist, t, b;
    int pred;  // index of predecessor label at previous position
    int via;   // station used on incoming arc (-1 none)
    int via2 = -1;  // optional second consecutive station (via -> via2 -> node)
};

// Evaluate sequence of customer node ids. If wantPath, reconstruct full path.
// prefix/startPos: reuse label sets L[0..startPos] of a route sharing the first startPos customers.
// saveTo: store all label sets (for later prefix reuse).
inline RouteEval evalRoute(const Instance& I, const std::vector<int>& seq, bool wantPath = false,
                           int maxLabels = 5, const std::vector<std::vector<Label>>* prefix = nullptr,
                           int startPos = 0, std::vector<std::vector<Label>>* saveTo = nullptr) {
    RouteEval res;
    double load = 0;
    for (int c : seq) load += I.nodes[c].demand;
    if (load > I.C + EPS) return res;
    const double depotDue = I.nodes[0].due;
    int m = seq.size();
    static thread_local std::vector<std::vector<Label>> L;
    if ((int)L.size() < m + 2) L.resize(m + 2);
    for (int i = 0; i < m + 2; i++) L[i].clear();
    if (prefix) {
        for (int i = 0; i <= startPos; i++) L[i] = (*prefix)[i];
    } else {
        startPos = 0;
        L[0].push_back({0, 0, I.Q, -1, -1});
    }
    auto tryAdd = [&](std::vector<Label>& vec, const Label& nl) {
        for (auto& o : vec)
            if (o.dist <= nl.dist + EPS && o.t <= nl.t + EPS && o.b >= nl.b - EPS) return;
        vec.erase(std::remove_if(vec.begin(), vec.end(),
                                 [&](const Label& o) {
                                     return nl.dist <= o.dist + EPS && nl.t <= o.t + EPS && nl.b >= o.b - EPS;
                                 }),
                  vec.end());
        vec.push_back(nl);
    };
    for (int pos = startPos; pos <= m; pos++) {
        int a = (pos == 0) ? 0 : seq[pos - 1];
        int bnode = (pos == m) ? 0 : seq[pos];
        const Node& B = I.nodes[bnode];
        bool endDepot = (pos == m);
        auto& out = L[pos + 1];
        for (int li = 0; li < (int)L[pos].size(); li++) {
            const Label& lb = L[pos][li];
            // arrival handling at bnode
            auto arriveAt = [&](double t, double bat, double dist, int via, int via2) {
                if (bat < -EPS) return;
                if (endDepot) {
                    if (t > depotDue + EPS) return;
                    tryAdd(out, {dist, t, bat, li, via, via2});
                    return;
                }
                if (t > B.due + EPS) return;
                double start = std::max(t, B.ready);
                double dep = start + B.service;
                // must be able to return to depot in time (direct travel lower bound)
                if (dep + I.d[bnode][0] / I.v > depotDue + EPS) return;
                tryAdd(out, {dist, dep, bat, li, via, via2});
            };
            // direct
            double dab = I.d[a][bnode];
            arriveAt(lb.t + dab / I.v, lb.b - I.r * dab, lb.dist + dab, -1, -1);
            // via one station
            for (int s : I.cand[a * I.n + bnode]) {
                double das = I.d[a][s];
                double bs = lb.b - I.r * das;
                if (bs < -EPS) continue;
                double ts = lb.t + das / I.v;
                if (ts > I.nodes[s].due + EPS) continue;
                double dep = ts + I.g * (I.Q - std::max(0.0, bs));
                double dsb = I.d[s][bnode];
                arriveAt(dep + dsb / I.v, I.Q - I.r * dsb, lb.dist + das + dsb, s, -1);
            }
            // via two consecutive stations, only when no direct or single-station option works for this label
            {
                int k1 = 0;
                for (int s : I.cand[a * I.n + bnode]) {
                    if (++k1 > I.M2) break;
                    double das = I.d[a][s];
                    double bs = lb.b - I.r * das;
                    if (bs < -EPS) continue;
                    double ts = lb.t + das / I.v;
                    if (ts > I.nodes[s].due + EPS) continue;
                    double dep = ts + I.g * (I.Q - std::max(0.0, bs));
                    int k2 = 0;
                    for (int s2 : I.cand[s * I.n + bnode]) {
                        if (++k2 > I.M2) break;
                        double d12 = I.d[s][s2];
                        if (I.Q - I.r * d12 < -EPS) continue;
                        double t2 = dep + d12 / I.v;
                        if (t2 > I.nodes[s2].due + EPS) continue;
                        double dep2 = t2 + I.g * (I.r * d12);
                        double d2b = I.d[s2][bnode];
                        arriveAt(dep2 + d2b / I.v, I.Q - I.r * d2b, lb.dist + das + d12 + d2b, s, s2);
                    }
                }
            }
        }
        if (out.empty()) { if (saveTo) saveTo->clear(); return res; }
        if ((int)out.size() > maxLabels) {
            std::sort(out.begin(), out.end(), [](const Label& x, const Label& y) { return x.dist < y.dist; });
            out.resize(maxLabels);
        }
    }
    int best = -1;
    for (int i = 0; i < (int)L[m + 1].size(); i++)
        if (best < 0 || L[m + 1][i].dist < L[m + 1][best].dist) best = i;
    res.feasible = true;
    res.dist = L[m + 1][best].dist;
    if (saveTo) saveTo->assign(L.begin(), L.begin() + m + 2);
    if (wantPath) {
        std::vector<int> rev;
        int li = best;
        for (int pos = m + 1; pos >= 1; pos--) {
            const Label& lb = L[pos][li];
            rev.push_back(pos == m + 1 ? 0 : seq[pos - 1]);
            if (lb.via2 >= 0) rev.push_back(lb.via2);
            if (lb.via >= 0) rev.push_back(lb.via);
            li = lb.pred;
        }
        rev.push_back(0);
        res.path.assign(rev.rbegin(), rev.rend());
    }
    return res;
}

// Quick necessary time-window check (no stations, no battery). Stations only delay, so failing here => infeasible.
inline bool quickTW(const Instance& I, const std::vector<int>& seq) {
    double t = 0;
    int prev = 0;
    for (int c : seq) {
        t += I.d[prev][c] / I.v;
        if (t > I.nodes[c].due + EPS) return false;
        t = std::max(t, I.nodes[c].ready) + I.nodes[c].service;
        prev = c;
    }
    t += I.d[prev][0] / I.v;
    return t <= I.nodes[0].due + EPS;
}

struct Solution {
    std::vector<std::vector<int>> routes;  // customer sequences
    std::vector<double> cost;              // per-route distance
    int unserved = 0;
    int nv() const {
        int k = 0;
        for (auto& r : routes) if (!r.empty()) k++;
        return k;
    }
    double td() const {
        double s = 0;
        for (size_t i = 0; i < routes.size(); i++) if (!routes[i].empty()) s += cost[i];
        return s;
    }
    // hierarchical fitness: vehicles first, then distance
    double fitness() const { return unserved * 1e9 + nv() * 1e5 + td(); }
    void compact() {
        std::vector<std::vector<int>> r2;
        std::vector<double> c2;
        for (size_t i = 0; i < routes.size(); i++)
            if (!routes[i].empty()) { r2.push_back(routes[i]); c2.push_back(cost[i]); }
        routes = r2;
        cost = c2;
    }
};
