#include "gutyansky_a_matrix_band_multiplication/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "gutyansky_a_matrix_band_multiplication/common/include/common.hpp"

namespace gutyansky_a_matrix_band_multiplication {

GutyanskyAMatrixBandMultiplicationSEQ::GutyanskyAMatrixBandMultiplicationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  GetInput() = in;
  GetOutput() = {};
}

bool GutyanskyAMatrixBandMultiplicationSEQ::ValidationImpl() {
  if (!GetInput().first.IsValid()) {
    return false;
  }
  if (!GetInput().second.IsValid()) {
    return false;
  }

  return GetInput().first.cols == GetInput().second.rows;
}

bool GutyanskyAMatrixBandMultiplicationSEQ::PreProcessingImpl() {
  return true;
}

bool GutyanskyAMatrixBandMultiplicationSEQ::RunImpl() {
  GetOutput().rows = GetInput().first.rows;
  GetOutput().cols = GetInput().second.cols;
  GetOutput().data.assign(GetOutput().rows * GetOutput().cols, 0);

  for (size_t i = 0; i < GetOutput().rows; i++) {
    for (size_t j = 0; j < GetOutput().cols; j++) {
      for (size_t k = 0; k < GetInput().first.cols; k++) {
        GetOutput().data[(i * GetOutput().cols) + j] += GetInput().first.data[(i * GetInput().first.cols) + k] *
                                                        GetInput().second.data[(k * GetInput().second.cols) + j];
      }
    }
  }

  return true;
}

bool GutyanskyAMatrixBandMultiplicationSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace gutyansky_a_matrix_band_multiplication
