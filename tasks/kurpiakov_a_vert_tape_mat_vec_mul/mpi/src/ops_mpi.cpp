#include "kurpiakov_a_vert_tape_mat_vec_mul/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "kurpiakov_a_vert_tape_mat_vec_mul/common/include/common.hpp"

namespace kurpiakov_a_vert_tape_mat_vec_mul {

KurpiakovAVretTapeMulMPI::KurpiakovAVretTapeMulMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
}

bool KurpiakovAVretTapeMulMPI::ValidationImpl() {
  if (rank_ != 0) {
    return true;
  }

  const auto &input = GetInput();
  const auto &size = std::get<0>(input);
  const auto &matrix = std::get<1>(input);
  const auto &vector = std::get<2>(input);

  if (size < 0) {
    return false;
  }

  if (size == 0) {
    return matrix.empty() && vector.empty();
  }

  auto expected_matrix_size = static_cast<size_t>(size) * static_cast<size_t>(size);
  if (matrix.size() != expected_matrix_size) {
    return false;
  }

  if (vector.size() != static_cast<size_t>(size)) {
    return false;
  }

  return true;
}

bool KurpiakovAVretTapeMulMPI::PreProcessingImpl() {
  if (rank_ == 0) {
    const auto &input = GetInput();
    const auto &size = std::get<0>(input);
    const auto &matrix = std::get<1>(input);
    const auto &vector = std::get<2>(input);
    matrix_size_ = size;
    matrix_data_ = matrix;
    vector_data_ = vector;
  }
  return true;
}

bool KurpiakovAVretTapeMulMPI::RunImpl() {
  MPI_Bcast(&matrix_size_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (matrix_size_ == 0) {
    GetOutput() = {};
    return true;
  }

  const int n = matrix_size_;

  vector_data_.resize(static_cast<size_t>(n));
  MPI_Bcast(vector_data_.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

  matrix_data_.resize(static_cast<size_t>(n) * static_cast<size_t>(n));
  MPI_Bcast(matrix_data_.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);

  const int base_cols = n / world_size_;
  const int extra_cols = n % world_size_;

  int col_start = 0;
  for (int i = 0; i < rank_; ++i) {
    col_start += base_cols + (i < extra_cols ? 1 : 0);
  }
  local_cols_ = base_cols + (rank_ < extra_cols ? 1 : 0);

  local_result_.assign(static_cast<size_t>(n), 0);

  for (int local_col = 0; local_col < local_cols_; ++local_col) {
    int global_col = col_start + local_col;
    int vec_val = vector_data_[global_col];

    for (int row = 0; row < n; ++row) {
      local_result_[row] +=
          matrix_data_[(static_cast<size_t>(row) * static_cast<size_t>(n)) + static_cast<size_t>(global_col)] * vec_val;
    }
  }

  result_.resize(static_cast<size_t>(n));
  MPI_Reduce(local_result_.data(), result_.data(), n, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(result_.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = result_;
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool KurpiakovAVretTapeMulMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kurpiakov_a_vert_tape_mat_vec_mul
