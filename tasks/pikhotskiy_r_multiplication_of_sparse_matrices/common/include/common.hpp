#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace pikhotskiy_r_multiplication_of_sparse_matrices {

struct SparseMatrixCRS {
  int rows;
  int cols;
  std::vector<double> values;
  std::vector<int> col_indices;
  std::vector<int> row_ptr;

  SparseMatrixCRS() : rows(0), cols(0) {}
  SparseMatrixCRS(int r, int c) : rows(r), cols(c), row_ptr(r + 1, 0) {}
};

using InType = std::tuple<SparseMatrixCRS, SparseMatrixCRS>;
using OutType = SparseMatrixCRS;
using TestType = std::tuple<SparseMatrixCRS, SparseMatrixCRS, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

SparseMatrixCRS DenseToCRS(const std::vector<double> &dense, int rows, int cols);

std::vector<double> CRSToDense(const SparseMatrixCRS &sparse);

SparseMatrixCRS TransposeCRS(const SparseMatrixCRS &matrix);

bool CompareSparseMatrices(const SparseMatrixCRS &a, const SparseMatrixCRS &b, double eps = 1e-9);

}  // namespace pikhotskiy_r_multiplication_of_sparse_matrices
