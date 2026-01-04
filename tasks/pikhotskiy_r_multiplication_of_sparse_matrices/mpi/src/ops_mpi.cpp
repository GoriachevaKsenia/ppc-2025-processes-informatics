#include "pikhotskiy_r_multiplication_of_sparse_matrices/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "pikhotskiy_r_multiplication_of_sparse_matrices/common/include/common.hpp"

namespace pikhotskiy_r_multiplication_of_sparse_matrices {

namespace {

double ComputeRowColProduct(const SparseMatrixCRS &mat_a, const SparseMatrixCRS &mat_bt, int row_a, int row_bt) {
  double sum = 0.0;
  int a_idx = mat_a.row_ptr[row_a];
  int a_end = mat_a.row_ptr[row_a + 1];
  int bt_idx = mat_bt.row_ptr[row_bt];
  int bt_end = mat_bt.row_ptr[row_bt + 1];

  while (a_idx < a_end && bt_idx < bt_end) {
    int a_col = mat_a.col_indices[a_idx];
    int bt_col = mat_bt.col_indices[bt_idx];
    if (a_col == bt_col) {
      sum += mat_a.values[a_idx] * mat_bt.values[bt_idx];
      ++a_idx;
      ++bt_idx;
    } else if (a_col < bt_col) {
      ++a_idx;
    } else {
      ++bt_idx;
    }
  }
  return sum;
}

void ComputeDisplacements(int size, const std::vector<int> &all_nnz, const std::vector<int> &all_num_rows,
                          std::vector<int> &nnz_displs, std::vector<int> &row_displs, int &total_nnz) {
  nnz_displs[0] = 0;
  row_displs[0] = 0;
  total_nnz = 0;
  for (int i = 0; i < size; ++i) {
    if (i > 0) {
      nnz_displs[i] = nnz_displs[i - 1] + all_nnz[i - 1];
      row_displs[i] = row_displs[i - 1] + all_num_rows[i - 1];
    }
    total_nnz += all_nnz[i];
  }
}

void BuildResultRowPtr(SparseMatrixCRS &result, int size, const std::vector<int> &all_nnz,
                       const std::vector<int> &all_num_rows, const std::vector<int> &all_row_ptr_shifted) {
  result.row_ptr[0] = 0;
  int current_offset = 0;
  int row_idx = 0;
  for (int proc = 0; proc < size; ++proc) {
    for (int ii = 0; ii < all_num_rows[proc]; ++ii) {
      result.row_ptr[row_idx + 1] = all_row_ptr_shifted[row_idx] + current_offset;
      ++row_idx;
    }
    current_offset += all_nnz[proc];
  }
}

}  // namespace

SparseMatrixMultiplicationMPI::SparseMatrixMultiplicationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }
}

void SparseMatrixMultiplicationMPI::BroadcastSparseMatrix(SparseMatrixCRS &matrix, int root) {
  MPI_Bcast(&matrix.rows, 1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&matrix.cols, 1, MPI_INT, root, MPI_COMM_WORLD);

  int nnz = static_cast<int>(matrix.values.size());
  MPI_Bcast(&nnz, 1, MPI_INT, root, MPI_COMM_WORLD);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != root) {
    matrix.values.resize(nnz);
    matrix.col_indices.resize(nnz);
    matrix.row_ptr.resize(matrix.rows + 1);
  }

  if (nnz > 0) {
    MPI_Bcast(matrix.values.data(), nnz, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(matrix.col_indices.data(), nnz, MPI_INT, root, MPI_COMM_WORLD);
  }
  MPI_Bcast(matrix.row_ptr.data(), matrix.rows + 1, MPI_INT, root, MPI_COMM_WORLD);
}

bool SparseMatrixMultiplicationMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int error_flag = 0;

  if (rank == 0) {
    const auto &mat_a = std::get<0>(GetInput());
    const auto &mat_b = std::get<1>(GetInput());

    if (mat_a.cols != mat_b.rows) {
      error_flag = 1;
    }
    if (mat_a.rows <= 0 || mat_a.cols <= 0 || mat_b.rows <= 0 || mat_b.cols <= 0) {
      error_flag = 1;
    }
    if (mat_a.row_ptr.size() != static_cast<std::size_t>(mat_a.rows) + 1) {
      error_flag = 1;
    }
    if (mat_b.row_ptr.size() != static_cast<std::size_t>(mat_b.rows) + 1) {
      error_flag = 1;
    }
  }

  MPI_Bcast(&error_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return error_flag == 0;
}

bool SparseMatrixMultiplicationMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    mat_a_ = std::get<0>(GetInput());
    mat_b_ = std::get<1>(GetInput());
    mat_b_transposed_ = TransposeCRS(mat_b_);
  }

  BroadcastSparseMatrix(mat_a_, 0);
  BroadcastSparseMatrix(mat_b_transposed_, 0);

  if (rank == 0) {
    mat_b_.rows = std::get<1>(GetInput()).rows;
    mat_b_.cols = std::get<1>(GetInput()).cols;
  }
  MPI_Bcast(&mat_b_.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&mat_b_.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

void SparseMatrixMultiplicationMPI::GatherResults(const SparseMatrixCRS &local_result, int my_num_rows) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int local_nnz = static_cast<int>(local_result.values.size());
  std::vector<int> all_nnz(size);
  std::vector<int> all_num_rows(size);
  MPI_Gather(&local_nnz, 1, MPI_INT, all_nnz.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&my_num_rows, 1, MPI_INT, all_num_rows.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> nnz_displs(size);
  std::vector<int> row_displs(size);
  int total_nnz = 0;
  if (rank == 0) {
    ComputeDisplacements(size, all_nnz, all_num_rows, nnz_displs, row_displs, total_nnz);
  }

  std::vector<double> all_values;
  std::vector<int> all_col_indices;
  if (rank == 0 && total_nnz > 0) {
    all_values.resize(total_nnz);
    all_col_indices.resize(total_nnz);
  }
  MPI_Gatherv(local_result.values.data(), local_nnz, MPI_DOUBLE, all_values.data(), all_nnz.data(), nnz_displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(local_result.col_indices.data(), local_nnz, MPI_INT, all_col_indices.data(), all_nnz.data(),
              nnz_displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_row_ptr_shifted(my_num_rows);
  for (int i = 0; i < my_num_rows; ++i) {
    local_row_ptr_shifted[i] = local_result.row_ptr[i + 1];
  }

  std::vector<int> all_row_ptr_shifted;
  if (rank == 0 && mat_a_.rows > 0) {
    all_row_ptr_shifted.resize(mat_a_.rows);
  }
  MPI_Gatherv(local_row_ptr_shifted.data(), my_num_rows, MPI_INT, all_row_ptr_shifted.data(), all_num_rows.data(),
              row_displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    SparseMatrixCRS result(mat_a_.rows, mat_b_.cols);
    result.values = std::move(all_values);
    result.col_indices = std::move(all_col_indices);
    result.row_ptr.resize(mat_a_.rows + 1);
    BuildResultRowPtr(result, size, all_nnz, all_num_rows, all_row_ptr_shifted);
    GetOutput() = result;
  }

  SparseMatrixCRS &output = GetOutput();
  BroadcastSparseMatrix(output, 0);
}

bool SparseMatrixMultiplicationMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_rows = mat_a_.rows;
  int base_rows = total_rows / size;
  int extra_rows = total_rows % size;

  int my_start_row = 0;
  for (int rr = 0; rr < rank; ++rr) {
    my_start_row += base_rows + (rr < extra_rows ? 1 : 0);
  }
  int my_num_rows = base_rows + (rank < extra_rows ? 1 : 0);

  SparseMatrixCRS local_result(my_num_rows, mat_b_.cols);
  local_result.row_ptr.resize(static_cast<std::size_t>(my_num_rows) + 1);
  if (!local_result.row_ptr.empty()) {
    local_result.row_ptr[0] = 0;
  }

  for (int local_i = 0; local_i < my_num_rows; ++local_i) {
    int global_i = my_start_row + local_i;
    for (int jj = 0; jj < mat_b_.cols; ++jj) {
      double sum = ComputeRowColProduct(mat_a_, mat_b_transposed_, global_i, jj);
      if (std::abs(sum) > 1e-12) {
        local_result.values.push_back(sum);
        local_result.col_indices.push_back(jj);
      }
    }
    local_result.row_ptr[local_i + 1] = static_cast<int>(local_result.values.size());
  }

  GatherResults(local_result, my_num_rows);
  return true;
}

bool SparseMatrixMultiplicationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace pikhotskiy_r_multiplication_of_sparse_matrices
