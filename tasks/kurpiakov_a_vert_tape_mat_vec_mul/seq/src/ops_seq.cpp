#include "kurpiakov_a_vert_tape_mat_vec_mul/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "kurpiakov_a_vert_tape_mat_vec_mul/common/include/common.hpp"

namespace kurpiakov_a_vert_tape_mat_vec_mul {

KurpiakovAVretTapeMulSEQ::KurpiakovAVretTapeMulSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool KurpiakovAVretTapeMulSEQ::ValidationImpl() {
  const auto &[size, matrix, vector] = GetInput();

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

bool KurpiakovAVretTapeMulSEQ::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

bool KurpiakovAVretTapeMulSEQ::RunImpl() {
  const auto total_size = std::get<0>(GetInput());

  if (total_size == 0) {
    GetOutput() = {};
    return true;
  }

  const auto &in_vec = std::get<2>(GetInput());
  const auto &in_mat = std::get<1>(GetInput());

  OutType res_vec(static_cast<size_t>(total_size), 0);
  for (int j = 0; j < total_size; j++) {
    for (int i = 0; i < total_size; i++) {
      res_vec[i] += in_mat[(i * total_size) + j] * in_vec[j];
    }
  }

  GetOutput() = res_vec;
  return true;
}

bool KurpiakovAVretTapeMulSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kurpiakov_a_vert_tape_mat_vec_mul
