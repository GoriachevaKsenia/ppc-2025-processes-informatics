#include "morozova_s_matrix_max_value/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "morozova_s_matrix_max_value/common/include/common.hpp"

namespace morozova_s_matrix_max_value {

MorozovaSMatrixMaxValueMPI::MorozovaSMatrixMaxValueMPI(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = 0;
}

bool MorozovaSMatrixMaxValueMPI::ValidationImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return true;
  }
  if (matrix[0].empty()) {
    return true;
  }
  const size_t cols = matrix[0].size();
  return std::ranges::all_of(matrix, [cols](const auto &row) { return row.size() == cols; });
}

bool MorozovaSMatrixMaxValueMPI::PreProcessingImpl() {
  return true;
}

bool MorozovaSMatrixMaxValueMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const auto &matrix = GetInput();
  if (matrix.empty() || matrix[0].empty()) {
    GetOutput() = 0;
    return true;
  }
  const int rows = static_cast<int>(matrix.size());
  const int cols = static_cast<int>(matrix[0].size());
  const int total = rows * cols;
  std::vector<int> flat;
  if (rank == 0) {
    flat.reserve(total);
    for (const auto &row : matrix) {
      flat.insert(flat.end(), row.begin(), row.end());
    }
  }
  std::vector<int> counts(size);
  std::vector<int> displs(size);
  const int base = total / size;
  const int rest = total % size;
  for (int i = 0, off = 0; i < size; ++i) {
    counts[i] = base + static_cast<int>(i < rest);
    displs[i] = off;
    off += counts[i];
  }
  std::vector<int> local(counts[rank]);
  MPI_Scatterv(rank == 0 ? flat.data() : nullptr, counts.data(), displs.data(), MPI_INT, local.data(), counts[rank],
               MPI_INT, 0, MPI_COMM_WORLD);

  int local_max = local[0];
  for (int v : local) {
    local_max = std::max(local_max, v);
  }
  int global_max = 0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  GetOutput() = global_max;
  return true;
}

bool MorozovaSMatrixMaxValueMPI::PostProcessingImpl() {
  return true;
}

}  // namespace morozova_s_matrix_max_value
