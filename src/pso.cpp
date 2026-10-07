// E-PSODS: Particle swarm optimisation with decoding schemes for the E-VRPTW
// Variants:
//   --decoder cos   : original PSODS-II rule (customer angle = 360*x_c, vehicle angle = 360*x_{N+k},
//                     preference by |cos(theta_c) - cos(theta_k)|), extended with charging-station insertion
//   --decoder geo   : proposed geo-anchored rule (customer angle = true polar angle around the depot,
//                     preference by circular angular distance to the vehicle angle)
//   --ls none|fcr|full : no local search, EV frontier-customer relocation/exchange (FCRLS-EV),
//                        or the same moves over all customers (full neighbourhood)
#include "common.h"
#include <chrono>
#include <random>

struct Params {
    std::string decoder = "geo";
    std::string ls = "fcr";
    int swarm = 30, iters = 150, seed = 1;
    double wmax = 0.9, wmin = 0.4, c1 = 1.5, c2 = 1.5, vmax = 0.2;
    int frontierArcs = 2;
    double lsDelta = 0.05;  // apply LS to a decoded solution if within (1+delta) of gbest (same NV)
    int insPositions = 4;   // positions evaluated exactly per route during insertion
    bool re = true;         // route elimination
    bool guideRaw = false;  // guide the swarm by raw decoded fitness instead of improved fitness
    bool lamarck = false;   // write the improved solution back into the position (approximate inverse decoding)
};

struct Solver {
    const Instance& I;
    Params P;
    std::mt19937_64 rng;
    int N, K;
    std::vector<double> geoAngle;  // per customer index 0..N-1
    long long evals = 0;

    Solver(const Instance& inst, Params p) : I(inst), P(p), rng(p.seed) {
        N = I.customers.size();
        double dem = 0;
        for (int c : I.customers) dem += I.nodes[c].demand;
        K = std::max(1, (int)std::ceil(dem / I.C - 1e-9));
        for (int c : I.customers) {
            double a = std::atan2(I.nodes[c].y - I.nodes[0].y, I.nodes[c].x - I.nodes[0].x) * 180.0 / M_PI;
            if (a < 0) a += 360;
            geoAngle.push_back(a);
        }
    }

    RouteEval eval(const std::vector<int>& seq) {
        evals++;
        return evalRoute(I, seq);
    }

    // Best feasible insertion of customer c into route seq. Returns (delta, position) or (INF,-1)
    std::pair<double, int> bestInsertion(const std::vector<int>& seq, double curCost, int c, double& newCost) {
        int m = seq.size();
        std::vector<std::pair<double, int>> est;
        for (int p = 0; p <= m; p++) {
            int a = p == 0 ? 0 : seq[p - 1];
            int b = p == m ? 0 : seq[p];
            est.push_back({I.d[a][c] + I.d[c][b] - I.d[a][b], p});
        }
        std::sort(est.begin(), est.end());
        double bestD = INF;
        int bestP = -1;
        int tried = 0;
        std::vector<int> tmp;
        // label sets of the current route, reused as prefix for every insertion position
        static thread_local std::vector<std::vector<Label>> cache;
        bool haveCache = false;
        for (auto& e : est) {
            if (tried >= P.insPositions) break;
            tmp = seq;
            tmp.insert(tmp.begin() + e.second, c);
            if (!quickTW(I, tmp)) continue;
            tried++;
            if (!haveCache) {
                if (seq.empty()) { cache.assign(1, {Label{0, 0, I.Q, -1, -1, -1}}); }
                else {
                    std::vector<int> seqNoLoad = seq;  // evaluate current route to fill cache
                    evals++;
                    RouteEval cur = evalRoute(I, seqNoLoad, false, 5, nullptr, 0, &cache);
                    if (!cur.feasible) cache.clear();
                }
                haveCache = true;
            }
            evals++;
            RouteEval re = cache.size() > (size_t)e.second ? evalRoute(I, tmp, false, 5, &cache, e.second)
                                                           : evalRoute(I, tmp);
            if (re.feasible && re.dist - curCost < bestD) {
                bestD = re.dist - curCost;
                bestP = e.second;
                newCost = re.dist;
            }
        }
        return {bestD, bestP};
    }

    Solution decode(const std::vector<double>& x) {
        // customer precedence list: ascending particle values of first N dimensions
        std::vector<int> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return x[a] < x[b]; });
        Solution S;
        S.routes.assign(K, {});
        S.cost.assign(K, 0.0);
        std::vector<int> failing;
        std::vector<std::pair<double, int>> pref(K);
        for (int ci : order) {
            int c = I.customers[ci];
            for (int k = 0; k < K; k++) {
                double vk = 360.0 * x[N + k];
                double score;
                if (P.decoder == "cos") {
                    double tc = 360.0 * x[ci];
                    score = std::fabs(std::cos(tc * M_PI / 180) - std::cos(vk * M_PI / 180));
                } else {
                    double dd = std::fabs(geoAngle[ci] - vk);
                    score = std::min(dd, 360 - dd);
                }
                pref[k] = {score, k};
            }
            std::sort(pref.begin(), pref.end());
            bool ok = false;
            for (auto& pk : pref) {
                int k = pk.second;
                double nc;
                auto ins = bestInsertion(S.routes[k], S.cost[k], c, nc);
                if (ins.second >= 0) {
                    S.routes[k].insert(S.routes[k].begin() + ins.second, c);
                    S.cost[k] = nc;
                    ok = true;
                    break;
                }
            }
            if (!ok) failing.push_back(c);
        }
        // reassign failing customers: best insertion over all routes, else open a new route
        for (int c : failing) {
            double best = INF, bestNc = 0;
            int bk = -1, bp = -1;
            for (size_t k = 0; k < S.routes.size(); k++) {
                double nc;
                auto ins = bestInsertion(S.routes[k], S.cost[k], c, nc);
                if (ins.second >= 0 && ins.first < best) { best = ins.first; bk = k; bp = ins.second; bestNc = nc; }
            }
            if (bk >= 0) {
                S.routes[bk].insert(S.routes[bk].begin() + bp, c);
                S.cost[bk] = bestNc;
            } else {
                std::vector<int> single{c};
                RouteEval re = eval(single);
                if (re.feasible) { S.routes.push_back(single); S.cost.push_back(re.dist); }
                else S.unserved++;
            }
        }
        S.compact();
        return S;
    }

    // Frontier customers of a route: endpoints (customers only) of its top-n longest arcs
    std::vector<int> frontier(const std::vector<int>& seq) {
        int m = seq.size();
        std::vector<std::pair<double, int>> arcs;
        for (int p = 0; p <= m; p++) {
            int a = p == 0 ? 0 : seq[p - 1];
            int b = p == m ? 0 : seq[p];
            arcs.push_back({-I.d[a][b], p});
        }
        std::sort(arcs.begin(), arcs.end());
        std::vector<int> fc;
        for (int i = 0; i < (int)arcs.size() && i < P.frontierArcs; i++) {
            int p = arcs[i].second;
            if (p > 0) fc.push_back(seq[p - 1]);
            if (p < m) fc.push_back(seq[p]);
        }
        std::sort(fc.begin(), fc.end());
        fc.erase(std::unique(fc.begin(), fc.end()), fc.end());
        return fc;
    }

    static std::vector<int> without(const std::vector<int>& s, int c) {
        std::vector<int> r;
        for (int x : s) if (x != c) r.push_back(x);
        return r;
    }

    // Inter-route relocate / exchange local search restricted to frontier customers (fcr) or all (full)
    void localSearch(Solution& S) {
        if (P.ls == "none") return;
        bool full = P.ls == "full";
        for (int pass = 0; pass < 100; pass++) {
            int R = S.routes.size();
            std::vector<std::vector<int>> cands(R);
            for (int r = 0; r < R; r++) cands[r] = full ? S.routes[r] : frontier(S.routes[r]);
            // best move of this pass
            double bestGain = 1e-7;
            int type = -1, r1b = -1, r2b = -1;
            std::vector<int> nr1, nr2;
            double nc1 = 0, nc2 = 0;
            for (int r1 = 0; r1 < R; r1++) {
                for (int f1 : cands[r1]) {
                    std::vector<int> r1w = without(S.routes[r1], f1);
                    double c1w = 0;
                    bool r1ok = true;
                    if (!r1w.empty()) {
                        RouteEval e = eval(r1w);
                        r1ok = e.feasible;
                        c1w = e.dist;
                    }
                    for (int r2 = 0; r2 < R; r2++) {
                        if (r2 == r1) {
                            // intra-route relocation of f1 to its best position in its own route
                            if (!r1ok || r1w.empty()) continue;
                            double nc;
                            auto ins = bestInsertion(r1w, c1w, f1, nc);
                            if (ins.second >= 0) {
                                double gain = S.cost[r1] - nc;
                                if (gain > bestGain) {
                                    bestGain = gain; type = 2; r1b = r1; r2b = -1;
                                    nr1 = r1w; nr1.insert(nr1.begin() + ins.second, f1);
                                    nc1 = nc;
                                }
                            }
                            continue;
                        }
                        // relocate f1 -> r2
                        if (r1ok) {
                            double nc;
                            auto ins = bestInsertion(S.routes[r2], S.cost[r2], f1, nc);
                            if (ins.second >= 0) {
                                double gain = (S.cost[r1] + S.cost[r2]) - (c1w + nc);
                                if (r1w.empty()) gain += 1e5;  // route eliminated: fleet reduction dominates
                                if (gain > bestGain) {
                                    bestGain = gain; type = 0; r1b = r1; r2b = r2;
                                    nr1 = r1w; nr2 = S.routes[r2];
                                    nr2.insert(nr2.begin() + ins.second, f1);
                                    nc1 = c1w; nc2 = nc;
                                }
                            }
                        }
                        // exchange f1 <-> f2 (each re-inserted at its best position)
                        if (r2 < r1) continue;  // symmetric
                        for (int f2 : cands[r2]) {
                            std::vector<int> r2w = without(S.routes[r2], f2);
                            if (r2w.empty() && r1w.empty()) continue;
                            double c2w = 0;
                            if (!r2w.empty()) {
                                if (!quickTW(I, r2w)) continue;
                                RouteEval e = eval(r2w);
                                if (!e.feasible) continue;
                                c2w = e.dist;
                            }
                            double ncA, ncB;
                            double baseA = r1w.empty() ? 0 : c1w;
                            auto insA = bestInsertion(r1w, baseA, f2, ncA);
                            if (insA.second < 0) continue;
                            auto insB = bestInsertion(r2w, c2w, f1, ncB);
                            if (insB.second < 0) continue;
                            double gain = (S.cost[r1] + S.cost[r2]) - (ncA + ncB);
                            if (gain > bestGain) {
                                bestGain = gain; type = 1; r1b = r1; r2b = r2;
                                nr1 = r1w; nr1.insert(nr1.begin() + insA.second, f2);
                                nr2 = r2w; nr2.insert(nr2.begin() + insB.second, f1);
                                nc1 = ncA; nc2 = ncB;
                            }
                        }
                    }
                }
            }
            if (type < 0) break;
            S.routes[r1b] = nr1; S.cost[r1b] = nc1;
            if (r2b >= 0) { S.routes[r2b] = nr2; S.cost[r2b] = nc2; }
            S.compact();
        }
    }

    // Route elimination: try to empty the shortest routes by re-inserting their customers elsewhere
    void routeElimination(Solution& S) {
        if (!P.re) return;
        bool improved = true;
        while (improved && S.routes.size() > 1) {
            improved = false;
            std::vector<int> idx(S.routes.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int a, int b) { return S.routes[a].size() < S.routes[b].size(); });
            for (int t = 0; t < (int)idx.size() && t < 3 && !improved; t++) {
                int r = idx[t];
                Solution T = S;
                std::vector<int> moving = T.routes[r];
                T.routes[r].clear();
                T.cost[r] = 0;
                bool ok = true;
                for (int c : moving) {
                    double best = INF, bestNc = 0;
                    int bk = -1, bp = -1;
                    for (size_t k = 0; k < T.routes.size(); k++) {
                        if ((int)k == r || T.routes[k].empty()) continue;
                        double nc;
                        auto ins = bestInsertion(T.routes[k], T.cost[k], c, nc);
                        if (ins.second >= 0 && ins.first < best) { best = ins.first; bk = k; bp = ins.second; bestNc = nc; }
                    }
                    if (bk < 0) { ok = false; break; }
                    T.routes[bk].insert(T.routes[bk].begin() + bp, c);
                    T.cost[bk] = bestNc;
                }
                if (ok) { T.compact(); S = T; improved = true; }
            }
        }
    }

    void intensify(Solution& S) {
        routeElimination(S);
        localSearch(S);
    }

    // Approximate inverse of the geo decoder: customer values follow the route order, vehicle angles are
    // the circular mean polar angle of each route (first K routes, taken in angular order)
    void encode(const Solution& S, std::vector<double>& x) {
        std::vector<int> idx(I.n, -1);
        for (int i = 0; i < N; i++) idx[I.customers[i]] = i;
        std::vector<std::pair<double, int>> ang;
        for (int r = 0; r < (int)S.routes.size(); r++) {
            double sx = 0, sy = 0;
            for (int c : S.routes[r]) { double a = geoAngle[idx[c]] * M_PI / 180; sx += std::cos(a); sy += std::sin(a); }
            double a = std::atan2(sy, sx) * 180 / M_PI; if (a < 0) a += 360;
            ang.push_back({a, r});
        }
        std::sort(ang.begin(), ang.end());
        int rank = 0;
        for (auto& ar : ang)
            for (int c : S.routes[ar.second]) x[idx[c]] = (rank++ + 0.5) / N;
        for (int k = 0; k < K && k < (int)ang.size(); k++) x[N + k] = ang[k].first / 360.0;
    }

    struct Result {
        Solution best;
        double seconds;
        std::vector<std::pair<int, double>> history;  // (iteration, best fitness)
    };

    Result run() {
        auto t0 = std::chrono::steady_clock::now();
        int D = N + K;
        std::uniform_real_distribution<double> U(0, 1);
        std::vector<std::vector<double>> X(P.swarm, std::vector<double>(D)), V = X, Pb = X;
        std::vector<double> pbF(P.swarm, INF);
        std::vector<Solution> pbS(P.swarm);
        Solution aS;
        double gF = INF, aF = INF;
        std::vector<double> G(D);
        Result res;
        for (int j = 0; j < P.swarm; j++)
            for (int d = 0; d < D; d++) { X[j][d] = U(rng); V[j][d] = (U(rng) - 0.5) * P.vmax; }
        for (int it = 0; it <= P.iters; it++) {
            for (int j = 0; j < P.swarm; j++) {
                Solution S = decode(X[j]);
                double f = S.fitness();
                const double fRaw = f;  // fitness of the decoded position itself
                // selective intensification: LS for promising particles (relative to the best solution found)
                if (aF >= INF || (S.nv() <= aS.nv() && S.td() <= aS.td() * (1 + P.lsDelta)) || S.nv() < aS.nv()) {
                    intensify(S);
                    f = S.fitness();
                }
                if (f < aF) { aF = f; aS = S; }  // archive of the best solution found (reported)
                double fg = P.guideRaw ? fRaw : f;
                if (P.lamarck && f < fRaw) {
                    // Lamarckian re-encoding: move the particle to a position that decodes close to S,
                    // then credit that position with what it actually decodes to
                    encode(S, X[j]);
                    Solution S2 = decode(X[j]);
                    fg = S2.fitness();
                    if (fg < aF) { aF = fg; aS = S2; }
                }
                if (fg < pbF[j]) { pbF[j] = fg; Pb[j] = X[j]; }
                if (fg < gF) { gF = fg; G = X[j]; }
            }
            if (it % 5 == 0 || it == P.iters) res.history.push_back({it, aF});
            if (it == P.iters) break;
            double w = P.wmax - (P.wmax - P.wmin) * it / (double)P.iters;
            for (int j = 0; j < P.swarm; j++)
                for (int d = 0; d < D; d++) {
                    double v = w * V[j][d] + P.c1 * U(rng) * (Pb[j][d] - X[j][d]) + P.c2 * U(rng) * (G[d] - X[j][d]);
                    v = std::max(-P.vmax, std::min(P.vmax, v));
                    double xn = X[j][d] + v;
                    if (xn > 1) { xn = 1; v = 0; }
                    if (xn < 0) { xn = 0; v = 0; }
                    X[j][d] = xn;
                    V[j][d] = v;
                }
        }
        res.best = aS;
        res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return res;
    }
};

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: pso instance [--decoder cos|geo] [--ls none|fcr|full] [--seed s] [--swarm n] [--iters n] [--out file]\n"); return 1; }
    Instance I;
    I.load(argv[1]);
    Params P;
    std::string out;
    for (int i = 2; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--decoder") P.decoder = v;
        else if (k == "--ls") P.ls = v;
        else if (k == "--seed") P.seed = atoi(v.c_str());
        else if (k == "--swarm") P.swarm = atoi(v.c_str());
        else if (k == "--iters") P.iters = atoi(v.c_str());
        else if (k == "--c") P.c1 = P.c2 = atof(v.c_str());
        else if (k == "--frontier") P.frontierArcs = atoi(v.c_str());
        else if (k == "--out") out = v;
        else if (k == "--re") P.re = atoi(v.c_str());
        else if (k == "--guide") { P.guideRaw = (v == "raw"); P.lamarck = (v == "lamarck"); }
    }
    Solver S(I, P);
    auto R = S.run();
    // output JSON
    std::ostringstream js;
    js << "{\"instance\":\"" << I.name << "\",\"decoder\":\"" << P.decoder << "\",\"ls\":\"" << P.ls
       << "\",\"re\":" << P.re << ",\"guide\":\"" << (P.lamarck ? "lamarck" : P.guideRaw ? "raw" : "improved") << "\"" << ",\"seed\":" << P.seed << ",\"swarm\":" << P.swarm << ",\"iters\":" << P.iters
       << ",\"K\":" << S.K << ",\"nv\":" << R.best.nv() << ",\"unserved\":" << R.best.unserved;
    char buf[64];
    snprintf(buf, sizeof buf, "%.4f", R.best.td());
    js << ",\"td\":" << buf;
    snprintf(buf, sizeof buf, "%.3f", R.seconds);
    js << ",\"seconds\":" << buf << ",\"evals\":" << S.evals << ",\"history\":[";
    for (size_t i = 0; i < R.history.size(); i++) {
        snprintf(buf, sizeof buf, "%.4f", R.history[i].second);
        js << (i ? "," : "") << "[" << R.history[i].first << "," << buf << "]";
    }
    js << "],\"routes\":[";
    for (size_t r = 0; r < R.best.routes.size(); r++) {
        RouteEval e = evalRoute(I, R.best.routes[r], true);
        js << (r ? "," : "") << "[";
        for (size_t q = 0; q < e.path.size(); q++) js << (q ? "," : "") << "\"" << I.nodes[e.path[q]].id << "\"";
        js << "]";
    }
    js << "]}";
    if (out.empty()) printf("%s\n", js.str().c_str());
    else { std::ofstream f(out); f << js.str() << "\n"; }
    fprintf(stderr, "%s %s/%s/re%d seed %d: NV=%d TD=%.2f unserved=%d time=%.1fs\n", I.name.c_str(), P.decoder.c_str(),
            P.ls.c_str(), (int)P.re, P.seed, R.best.nv(), R.best.td(), R.best.unserved, R.seconds);
    return 0;
}
