#pragma once

#include <cstddef>
#include <vector>

#include "task/include/task.hpp"

namespace matrix_band_multiplication {

struct Matrix {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<double> values;
};

struct MatrixMulInput {
  Matrix a;
  Matrix b;
};

using InType = MatrixMulInput;
using OutType = Matrix;
using BaseTask = ppc::task::Task<InType, OutType>;

inline std::size_t FlattenIndex(std::size_t row, std::size_t col, std::size_t cols) {
  const std::size_t row_offset = row * cols;
  return row_offset + col;
}

}  // namespace matrix_band_multiplication
