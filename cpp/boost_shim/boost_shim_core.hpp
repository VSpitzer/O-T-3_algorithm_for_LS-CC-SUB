// Minimal stand-in for the four Boost.Graph facilities solve_CLSP_IB.cpp uses.
//
// It exists so the C++ can be built and validated on a machine without Boost.
// Semantics deliberately mirror Boost: the no-root bellman_ford requires the
// caller to have initialised the distance map, relaxes all edges up to |V|-1
// times, records predecessors, and returns false on a negative cycle.
//
// NOT a general Boost.Graph replacement -- only the pieces this project needs.
// For anything beyond building/validating solve_CLSP_IB.cpp, install real Boost.
#pragma once
#include <vector>
#include <cstddef>
#include <limits>
#include <utility>
#include <cassert>
#include <functional>
#include <mutex>

namespace boost {

struct directedS {}; struct vecS {}; struct no_property {};
struct edge_weight_t {}; struct vertex_distance_t {};
inline constexpr edge_weight_t edge_weight{};
inline constexpr vertex_distance_t vertex_distance{};
template <class Tag, class T, class Next = no_property> struct property { typedef T value_type; };

struct edges_are_unsorted_multi_pass_t {};
inline constexpr edges_are_unsorted_multi_pass_t edges_are_unsorted_multi_pass{};

struct ShimGraph {
    std::size_t n = 0;
    std::vector<std::size_t> src, tgt;
    std::vector<float> w;
    ShimGraph() = default;
    explicit ShimGraph(std::size_t nv) : n(nv) {}
    template <class EI, class WI>
    ShimGraph(edges_are_unsorted_multi_pass_t, EI first, EI last, WI wf, std::size_t nv) : n(nv) {
        for (EI it = first; it != last; ++it, ++wf) {
            src.push_back(static_cast<std::size_t>(it->first));
            tgt.push_back(static_cast<std::size_t>(it->second));
            w.push_back(static_cast<float>(*wf));
        }
    }
};

template <class A=void,class B=void,class C=void,class D=void,class E=void>
struct adjacency_list : ShimGraph { using ShimGraph::ShimGraph; };
template <class A=void,class B=void,class C=void>
struct compressed_sparse_row_graph : ShimGraph { using ShimGraph::ShimGraph; };

template <class G> struct graph_traits {
    typedef std::size_t vertex_descriptor;
    typedef std::size_t edge_descriptor;
    static vertex_descriptor null_vertex() { return (std::numeric_limits<std::size_t>::max)(); }
};

inline std::size_t num_vertices(const ShimGraph& g) { return g.n; }

// Serialised: any heap corruption observed is then the caller's, not the shim's.
// Real Boost pre-sizes the vertex array and keeps disjoint per-vertex out-edge
// lists, so it does not have the shared-vector growth this would otherwise have.
inline std::mutex& shim_edge_mutex() { static std::mutex m; return m; }
inline void add_edge(std::size_t u, std::size_t v, float wt, ShimGraph& g) {
    std::lock_guard<std::mutex> lk(shim_edge_mutex());
    if (u + 1 > g.n) g.n = u + 1;
    if (v + 1 > g.n) g.n = v + 1;
    g.src.push_back(u); g.tgt.push_back(v); g.w.push_back(wt);
}

struct WeightMapTag {};
inline WeightMapTag get(edge_weight_t, const ShimGraph&) { return {}; }

struct BFParams {
    float* dist = nullptr;
    std::function<void(std::size_t, std::size_t)> setpred;
    template <class P> BFParams& predecessor_map(P* p) {
        setpred = [p](std::size_t v, std::size_t u) { p[v] = static_cast<P>(u); };
        return *this;
    }
    BFParams& distance_map(float* d) { dist = d; return *this; }
};
inline BFParams weight_map(WeightMapTag) { return BFParams{}; }

inline bool bellman_ford_shortest_paths(const ShimGraph& g, const BFParams& p) {
    const float INF = std::numeric_limits<float>::infinity();
    const std::size_t n = g.n, m = g.src.size();
    for (std::size_t pass = 0; pass + 1 < n; ++pass) {
        bool changed = false;
        for (std::size_t e = 0; e < m; ++e) {
            std::size_t u = g.src[e], v = g.tgt[e];
            if (p.dist[u] == INF) continue;
            float nd = p.dist[u] + g.w[e];
            if (nd < p.dist[v]) { p.dist[v] = nd; if (p.setpred) p.setpred(v, u); changed = true; }
        }
        if (!changed) return true;
    }
    for (std::size_t e = 0; e < m; ++e) {
        std::size_t u = g.src[e], v = g.tgt[e];
        if (p.dist[u] != INF && p.dist[u] + g.w[e] < p.dist[v]) return false;
    }
    return true;
}

} // namespace boost
