#include "kondakov_v_reduce/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>

#include "kondakov_v_reduce/common/include/common.hpp"

namespace kondakov_v_reduce {

KondakovVReduceTaskMPI::KondakovVReduceTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KondakovVReduceTaskMPI::ValidationImpl() {
  return true;
}

bool KondakovVReduceTaskMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool KondakovVReduceTaskMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &values = GetInput().values;
  const ReduceOp op = GetInput().op;
  const std::size_t n = values.size();

  std::size_t block_size = n / size;
  std::size_t remainder = n % size;
  auto r = static_cast<std::size_t>(rank);

  std::size_t local_start = (r * block_size) + std::min(r, remainder);
  std::size_t local_end = local_start + block_size + (r < remainder ? 1 : 0);

  OutType local_result = 0;
  bool has_data = (local_start < local_end);

  if (has_data) {
    local_result = values[local_start];
    for (std::size_t i = local_start + 1; i < local_end; ++i) {
      local_result = ApplyReduceOp(local_result, values[i], op);
    }
  } else {
    local_result = GetNeutralElement(op);
  }

  OutType current = local_result;
  int mask = 1;

  while (mask < size) {
    int partner = rank ^ mask;
    if (partner < size) {
      if ((rank & mask) == 0) {
        OutType received = 0;
        MPI_Recv(&received, 1, MPI_INT64_T, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        current = ApplyReduceOp(current, received, op);
      } else {
        MPI_Send(&current, 1, MPI_INT64_T, partner, 0, MPI_COMM_WORLD);
        break;
      }
    }
    mask <<= 1;
  }

  if (rank == 0) {
    GetOutput() = current;
  }

  return true;
}

bool KondakovVReduceTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kondakov_v_reduce
