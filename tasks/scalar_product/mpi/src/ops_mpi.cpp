#include "scalar_product/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <numeric>
#include <vector>

#include "scalar_product/common/include/common.hpp"

namespace scalar_product {

ScalarProductMPI::ScalarProductMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ScalarProductMPI::ValidationImpl() {
  const auto &vector_a = GetInput().first;
  const auto &vector_b = GetInput().second;
  return vector_a.size() == vector_b.size() && !vector_a.empty();
}

bool ScalarProductMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  const auto &vector_a = GetInput().first;
  const auto &vector_b = GetInput().second;

  int global_size = 0;
  if (rank_ == 0) {
    global_size = static_cast<int>(vector_a.size());
  }
  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> counts(world_size_);
  std::vector<int> displase(world_size_, 0);

  const int base = global_size / world_size_;
  int remain = global_size % world_size_;

  for (int i = 0; i < world_size_; ++i) {
    counts[i] = base + (i < remain ? 1 : 0);
  }

  for (int i = 1; i < world_size_; ++i) {
    displase[i] = displase[i - 1] + counts[i - 1];
  }
  const int local_count = counts[rank_];

  local_vector_a_.assign(static_cast<std::size_t>(local_count), 0);
  local_vector_b_.assign(static_cast<std::size_t>(local_count), 0);

  MPI_Scatterv(rank_ == 0 ? vector_a.data() : nullptr, counts.data(), displase.data(), MPI_INT, local_vector_a_.data(),
               local_count, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(rank_ == 0 ? vector_b.data() : nullptr, counts.data(), displase.data(), MPI_INT, local_vector_b_.data(),
               local_count, MPI_INT, 0, MPI_COMM_WORLD);

  local_sum_ = 0;
  result_ = 0;
  return true;
}

bool ScalarProductMPI::RunImpl() {
  local_sum_ = std::inner_product(local_vector_a_.begin(), local_vector_a_.end(), local_vector_b_.begin(), 0);
  return MPI_Reduce(&local_sum_, &result_, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD) == MPI_SUCCESS;
}

bool ScalarProductMPI::PostProcessingImpl() {
  MPI_Bcast(&result_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = result_;
  return true;
}

}  // namespace scalar_product
