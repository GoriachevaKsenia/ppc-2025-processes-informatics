#pragma once

#include <mpi.h>

#include <cstddef>
#include <vector>

namespace shkrebko_m_shell_sort_batcher_merge {

inline std::vector<int> RecvVector(int src, int tag_base, MPI_Comm comm) {
  int sz = 0;
  MPI_Status status{};
  MPI_Recv(&sz, 1, MPI_INT, src, tag_base, comm, &status);

  std::vector<int> v(static_cast<std::size_t>(sz));
  if (sz > 0) {
    MPI_Recv(v.data(), sz, MPI_INT, src, tag_base + 1, comm, &status);
  }
  return v;
}

inline void SendVector(int dst, int tag_base, const std::vector<int> &v, MPI_Comm comm) {
  const int sz = static_cast<int>(v.size());
  MPI_Send(&sz, 1, MPI_INT, dst, tag_base, comm);
  if (sz > 0) {
    MPI_Send(v.data(), sz, MPI_INT, dst, tag_base + 1, comm);
  }
}

}  // namespace shkrebko_m_shell_sort_batcher_merge
