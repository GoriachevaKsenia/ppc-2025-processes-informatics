#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "task/include/task.hpp"

namespace potashnik_m_short_ways_bellford {

// CRS Graph class
class Graph {
 public:
  int n;

  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<int> weights;
  Graph() : n(0) {}
  explicit Graph(int n_vertices) : n(n_vertices), row_ptr(n_vertices + 1, 0) {}

  void BuildGraph(const std::vector<int> &src, const std::vector<int> &dst, const std::vector<int> &w) {
    int m = static_cast<int>(src.size());

    for (int i = 0; i < m; i++) {
      row_ptr[src[i] + 1]++;
    }

    for (int i = 0; i < n; i++) {
      row_ptr[i + 1] += row_ptr[i];
    }

    col_ind.resize(m);
    weights.resize(m);
    std::vector<int> cur = row_ptr;

    for (int i = 0; i < m; i++) {
      int u = src[i];
      int pos = cur[u]++;

      col_ind[pos] = dst[i];
      weights[pos] = w[i];
    }
  }

  [[nodiscard]] int Begin(int u) const {
    return row_ptr[u];
  }
  [[nodiscard]] int End(int u) const {
    return row_ptr[u + 1];
  }
};

inline void IterateThroughVertex(const Graph &g, int u, const std::vector<int> &dist, std::vector<int> &dist_out) {
  for (int i = g.Begin(u); i < g.End(u); i++) {
    int v = g.col_ind[i];
    int w = g.weights[i];

    int new_dist = dist[u] + w;
    dist_out[v] = std::min(new_dist, dist_out[v]);
  }
}

inline Graph GenerateGraph(int n) {
  Graph g(n);
  std::vector<int> src;
  std::vector<int> dst;
  std::vector<int> w;
  int layers = static_cast<int>(std::sqrt(n));
  layers = std::max(layers, 1);
  int layer_size = n / layers;
  for (int lidx = 0; lidx < layers - 1; lidx++) {
    int start_u = lidx * layer_size;
    int end_u = (lidx + 1) * layer_size;
    int start_v = (lidx + 1) * layer_size;
    int end_v = (lidx + 2) * layer_size;
    end_v = std::min(end_v, n);
    for (int uidx = start_u; uidx < end_u; uidx++) {
      for (int vidx = start_v; vidx < end_v; vidx++) {
        src.push_back(uidx);
        dst.push_back(vidx);
        int weight = ((uidx * 13 + vidx * 7) % 10) + 1;
        w.push_back(weight);
      }
    }
  }
  g.BuildGraph(src, dst, w);
  return g;
}

using InType = Graph;              // Graph
using OutType = std::vector<int>;  // Vector of shortest paths
using TestType = int;              // Amount of vertices
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace potashnik_m_short_ways_bellford
