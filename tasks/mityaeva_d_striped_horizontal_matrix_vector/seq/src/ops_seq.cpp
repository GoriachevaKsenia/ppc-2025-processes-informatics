#include "mityaeva_d_striped_horizontal_matrix_vector/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "mityaeva_d_striped_horizontal_matrix_vector/common/include/common.hpp"

namespace mityaeva_d_striped_horizontal_matrix_vector {

StripedHorizontalMatrixVectorSEQ::StripedHorizontalMatrixVectorSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>{0.0};
}

bool StripedHorizontalMatrixVectorSEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.empty() || input.size() < 3) {
    return false;
  }

  rows_ = static_cast<int>(input[0]);
  cols_ = static_cast<int>(input[1]);

  if (rows_ <= 0 || cols_ <= 0) {
    return false;
  }

  if (static_cast<int>(input[2]) != cols_) {
    return false;
  }

  size_t expected_size = 3 + (static_cast<size_t>(rows_) * static_cast<size_t>(cols_)) + static_cast<size_t>(cols_);

  return input.size() == expected_size;
}

bool StripedHorizontalMatrixVectorSEQ::PreProcessingImpl() {
  return true;
}

bool StripedHorizontalMatrixVectorSEQ::RunImpl() {
  const auto &input = GetInput();

  try {
    size_t matrix_start_idx = 3;
    size_t vector_start_idx = matrix_start_idx + (static_cast<size_t>(rows_) * static_cast<size_t>(cols_));

    std::vector<double> result(rows_, 0.0);

    for (int i = 0; i < rows_; ++i) {
      double sum = 0.0;
      size_t row_start = matrix_start_idx + (static_cast<size_t>(i) * static_cast<size_t>(cols_));

      for (int j = 0; j < cols_; ++j) {
        double matrix_element = input[row_start + j];
        double vector_element = input[vector_start_idx + j];
        sum += matrix_element * vector_element;
      }

      result[i] = sum;
    }

    auto &output = GetOutput();
    output.clear();
    output.reserve(rows_ + 1);

    output.push_back(static_cast<double>(result.size()));

    for (const auto &val : result) {
      output.push_back(val);
    }

    return true;

  } catch (...) {
    return false;
  }
}

bool StripedHorizontalMatrixVectorSEQ::PostProcessingImpl() {
  const auto &output = GetOutput();

  if (output.empty()) {
    return false;
  }

  int result_size = static_cast<int>(output[0]);
  return result_size == rows_ && output.size() == static_cast<size_t>(result_size) + 1;
}

}  // namespace mityaeva_d_striped_horizontal_matrix_vector
