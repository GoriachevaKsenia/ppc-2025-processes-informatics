#include "morozova_s_matrix_max_value/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "morozova_s_matrix_max_value/common/include/common.hpp"

namespace morozova_s_matrix_max_value {
MorozovaSMatrixMaxValueSEQ::MorozovaSMatrixMaxValueSEQ(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = 0;
}

bool MorozovaSMatrixMaxValueSEQ::ValidationImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty() || matrix[0].empty()) {
    return true;
  }
  const size_t cols = matrix[0].size();
  bool all_equal = std::ranges::all_of(matrix, [cols](const auto &row) { return row.size() == cols; });
  if (!all_equal) {
    return true;
  }
  return true;
}

bool MorozovaSMatrixMaxValueSEQ::PreProcessingImpl() {
  return true;
}

bool MorozovaSMatrixMaxValueSEQ::RunImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty() || matrix[0].empty()) {
    GetOutput() = 0;
    return true;
  }
  const size_t cols = matrix[0].size();
  if (!std::ranges::all_of(matrix, [cols](const auto &row) { return row.size() == cols; })) {
    GetOutput() = 0;
    return true;
  }
  int max_value = matrix[0][0];
  for (const auto &row : matrix) {
    for (int value : row) {
      max_value = std::max(max_value, value);
    }
  }
  GetOutput() = max_value;
  return true;
}

bool MorozovaSMatrixMaxValueSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace morozova_s_matrix_max_value
