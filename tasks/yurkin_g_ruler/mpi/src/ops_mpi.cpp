#include "yurkin_g_ruler/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>

#include "yurkin_g_ruler/common/include/common.hpp"

namespace yurkin_g_ruler {

YurkinGRulerMPI::YurkinGRulerMPI(const InType &in) {
  SetTypeOfTask(YurkinGRulerMPI::GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YurkinGRulerMPI::ValidationImpl() {
  return (GetInput() >= 0) && (GetOutput() == 0);
}

bool YurkinGRulerMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YurkinGRulerMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size <= 0) {
    return false;
  }

  int src = 0;
  int dst = size - 1;

  src = std::clamp(src, 0, size - 1);
  dst = std::clamp(dst, 0, size - 1);

  GetOutput() = GetInput();
  int payload = GetInput();

  if (src == dst) {
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  const int low = std::min(src, dst);
  const int high = std::max(src, dst);
  const int direction = (dst > src) ? +1 : -1;

  if (rank < low || rank > high) {
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  if (rank == src) {
    const int next = rank + direction;
    MPI_Send(&payload, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  const int prev = rank - direction;
  int recv_val = 0;
  MPI_Recv(&recv_val, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (rank == dst) {
    GetOutput() = recv_val;
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  const int next = rank + direction;
  MPI_Send(&recv_val, 1, MPI_INT, next, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool YurkinGRulerMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace yurkin_g_ruler
