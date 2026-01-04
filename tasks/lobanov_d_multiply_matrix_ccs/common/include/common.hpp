#pragma once

#include <cstddef>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace lobanov_d_multiply_matrix_ccs {

struct CompressedColumnMatrix {
  int row_count = 0;
  int column_count = 0;
  int non_zero_count = 0;
  std::vector<double> value_data;
  std::vector<int> row_index_data;
  std::vector<int> column_pointer_data;

  CompressedColumnMatrix() = default;

  CompressedColumnMatrix(int r, int c, int nz) : row_count(r), column_count(c), non_zero_count(nz) {
    if (nz > 0) {
      value_data.reserve(static_cast<std::size_t>(nz));
      row_index_data.reserve(static_cast<std::size_t>(nz));
    }
    column_pointer_data.reserve(static_cast<std::size_t>(c) + 1);
  }

  CompressedColumnMatrix(const CompressedColumnMatrix &other) = default;
  CompressedColumnMatrix &operator=(const CompressedColumnMatrix &other) = default;

  ~CompressedColumnMatrix() = default;

  void ZeroInitialize() {
    row_count = 0;
    column_count = 0;
    non_zero_count = 0;
    value_data.clear();
    row_index_data.clear();
    column_pointer_data.clear();
  }

  [[nodiscard]] bool IsValid() const {
    if (row_count < 0 || column_count < 0 || non_zero_count < 0) {
      return false;
    }
    if (non_zero_count > 0) {
      if (value_data.size() != static_cast<std::size_t>(non_zero_count)) {
        return false;
      }
      if (row_index_data.size() != static_cast<std::size_t>(non_zero_count)) {
        return false;
      }
    }
    if (column_pointer_data.size() != static_cast<std::size_t>(column_count) + 1U) {
      return false;
    }
    if (!column_pointer_data.empty() && column_pointer_data[0] != 0) {
      return false;
    }
    return true;
  }
};

using InType = std::pair<CompressedColumnMatrix, CompressedColumnMatrix>;
using OutType = CompressedColumnMatrix;
using TestType = std::tuple<std::string, CompressedColumnMatrix, CompressedColumnMatrix, CompressedColumnMatrix>;
using BaseTask = ppc::task::Task<InType, OutType>;

CompressedColumnMatrix CreateRandomCompressedColumnMatrix(int row_count, int column_count, double density_factor,
                                                          int seed);

inline std::ostream &operator<<(std::ostream &os, const CompressedColumnMatrix &matrix) {
  os << "CompressedColumnMatrix{"
     << "rows=" << matrix.row_count << ", cols=" << matrix.column_count << ", nnz=" << matrix.non_zero_count << "}";
  return os;
}
}  // namespace lobanov_d_multiply_matrix_ccs
