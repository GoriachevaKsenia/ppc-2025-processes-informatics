#pragma once

#include <mpi.h>

#include <vector>

#include "potashnik_m_short_ways_bellford/common/include/common.hpp"
#include "task/include/task.hpp"

namespace potashnik_m_short_ways_bellford {

class PotashnikMShortWaysBellfordMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PotashnikMShortWaysBellfordMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

inline void BellmanFordAlgoIterationMpi(const Graph &g, const std::vector<int> &dist, std::vector<int> &dist_next,
                                        int start, int end) {
  dist_next = dist;
  for (int uidx = start; uidx < end; uidx++) {
    if (dist[uidx] == 1e9) {
      continue;
    }
    IterateThroughVertex(g, uidx, dist, dist_next);
  }
}

inline void BellmanFordAlgoMpi(const Graph &g, int source, std::vector<int> &dist) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = g.n;

  dist.assign(n, 1e9);
  if (rank == 0) {
    dist[source] = 0;
  }

  MPI_Bcast(dist.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> dist_next(n);

  int start = rank * n / size;
  int end = (rank + 1) * n / size;

  for (int i = 0; i < n - 1; i++) {
    BellmanFordAlgoIterationMpi(g, dist, dist_next, start, end);
    MPI_Allreduce(dist_next.data(), dist.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  }
}

}  // namespace potashnik_m_short_ways_bellford
