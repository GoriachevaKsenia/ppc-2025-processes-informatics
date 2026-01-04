#include "pikhotskiy_r_multiplication_of_sparse_matrices/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
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

}  // namespace

SparseMatrixCRS DenseToCRS(const std::vector<double> &dense, int rows, int cols) {
  SparseMatrixCRS result(rows, cols);
  result.row_ptr.resize(rows + 1);
  result.row_ptr[0] = 0;

  for (int i = 0; i < rows; ++i) {
    for (int jj = 0; jj < cols; ++jj) {
      double val = dense[(i * cols) + jj];
      if (std::abs(val) > 1e-12) {
        result.values.push_back(val);
        result.col_indices.push_back(jj);
      }
    }
    result.row_ptr[i + 1] = static_cast<int>(result.values.size());
  }
  return result;
}

std::vector<double> CRSToDense(const SparseMatrixCRS &sparse) {
  std::vector<double> dense(static_cast<std::size_t>(sparse.rows) * sparse.cols, 0.0);
  for (int i = 0; i < sparse.rows; ++i) {
    for (int kk = sparse.row_ptr[i]; kk < sparse.row_ptr[i + 1]; ++kk) {
      dense[(i * sparse.cols) + sparse.col_indices[kk]] = sparse.values[kk];
    }
  }
  return dense;
}

SparseMatrixCRS TransposeCRS(const SparseMatrixCRS &matrix) {
  SparseMatrixCRS result(matrix.cols, matrix.rows);
  result.row_ptr.resize(matrix.cols + 1, 0);

  for (int col : matrix.col_indices) {
    result.row_ptr[col + 1]++;
  }

  for (int i = 1; i <= matrix.cols; ++i) {
    result.row_ptr[i] += result.row_ptr[i - 1];
  }

  result.values.resize(matrix.values.size());
  result.col_indices.resize(matrix.col_indices.size());

  std::vector<int> current_pos(result.row_ptr.begin(), result.row_ptr.end() - 1);

  for (int i = 0; i < matrix.rows; ++i) {
    for (int kk = matrix.row_ptr[i]; kk < matrix.row_ptr[i + 1]; ++kk) {
      int col = matrix.col_indices[kk];
      int pos = current_pos[col]++;
      result.values[pos] = matrix.values[kk];
      result.col_indices[pos] = i;
    }
  }

  return result;
}

bool CompareSparseMatrices(const SparseMatrixCRS &aa, const SparseMatrixCRS &bb, double eps) {
  if (aa.rows != bb.rows || aa.cols != bb.cols) {
    return false;
  }
  auto dense_a = CRSToDense(aa);
  auto dense_b = CRSToDense(bb);
  for (std::size_t i = 0; i < dense_a.size(); ++i) {
    if (std::abs(dense_a[i] - dense_b[i]) > eps) {
      return false;
    }
  }
  return true;
}

SparseMatrixMultiplicationSEQ::SparseMatrixMultiplicationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SparseMatrixMultiplicationSEQ::ValidationImpl() {
  const auto &mat_a = std::get<0>(GetInput());
  const auto &mat_b = std::get<1>(GetInput());

  if (mat_a.cols != mat_b.rows) {
    return false;
  }
  if (mat_a.rows <= 0 || mat_a.cols <= 0 || mat_b.rows <= 0 || mat_b.cols <= 0) {
    return false;
  }
  if (mat_a.row_ptr.size() != static_cast<std::size_t>(mat_a.rows) + 1) {
    return false;
  }
  if (mat_b.row_ptr.size() != static_cast<std::size_t>(mat_b.rows) + 1) {
    return false;
  }
  return true;
}

bool SparseMatrixMultiplicationSEQ::PreProcessingImpl() {
  mat_a_ = std::get<0>(GetInput());
  mat_b_transposed_ = TransposeCRS(std::get<1>(GetInput()));
  return true;
}

bool SparseMatrixMultiplicationSEQ::RunImpl() {
  const auto &mat_b = std::get<1>(GetInput());
  SparseMatrixCRS result(mat_a_.rows, mat_b.cols);
  result.row_ptr.resize(static_cast<std::size_t>(mat_a_.rows) + 1);
  if (!result.row_ptr.empty()) {
    result.row_ptr[0] = 0;
  }

  for (int i = 0; i < mat_a_.rows; ++i) {
    for (int jj = 0; jj < mat_b.cols; ++jj) {
      double sum = ComputeRowColProduct(mat_a_, mat_b_transposed_, i, jj);
      if (std::abs(sum) > 1e-12) {
        result.values.push_back(sum);
        result.col_indices.push_back(jj);
      }
    }
    result.row_ptr[i + 1] = static_cast<int>(result.values.size());
  }

  GetOutput() = result;
  return true;
}

bool SparseMatrixMultiplicationSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace pikhotskiy_r_multiplication_of_sparse_matrices
