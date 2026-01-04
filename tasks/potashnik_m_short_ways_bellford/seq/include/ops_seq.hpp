#pragma once

#include <vector>

#include "potashnik_m_short_ways_bellford/common/include/common.hpp"
#include "task/include/task.hpp"

namespace potashnik_m_short_ways_bellford {

class PotashnikMShortWaysBellfordSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PotashnikMShortWaysBellfordSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

inline void BellmanFordAlgoIterationSeq(const Graph &g, const std::vector<int> &dist, std::vector<int> &dist_next) {
  int n = g.n;
  dist_next = dist;
  for (int uidx = 0; uidx < n; uidx++) {
    if (dist[uidx] == 1e9) {
      continue;
    }
    IterateThroughVertex(g, uidx, dist, dist_next);
  }
}

inline void BellmanFordAlgoSeq(const Graph &g, int source, std::vector<int> &dist) {
  int n = g.n;

  // Compiler wont let it work without this lines (This should be just a warning, but I can't not fix it with this)
  if (n == 0) {
    return;
  }
  if (source < 0 || source >= n) {
    return;
  }

  dist.assign(n, 1e9);
  dist[source] = 0;

  std::vector<int> dist_next(n);

  for (int i = 0; i < n - 1; i++) {
    BellmanFordAlgoIterationSeq(g, dist, dist_next);
    dist.swap(dist_next);
  }
}

}  // namespace potashnik_m_short_ways_bellford
