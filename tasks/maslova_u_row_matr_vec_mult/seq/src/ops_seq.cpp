#include "maslova_u_row_matr_vec_mult/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "maslova_u_row_matr_vec_mult/common/include/common.hpp"

namespace maslova_u_row_matr_vec_mult {

MaslovaURowMatrVecMultSEQ::MaslovaURowMatrVecMultSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MaslovaURowMatrVecMultSEQ::ValidationImpl() {
  const auto &matrix = GetInput().first;
  const auto &vec = GetInput().second;
  if (matrix.data.empty()) {
    return true;
  }
  return (matrix.cols == vec.size()) && (matrix.data.size() == matrix.rows * matrix.cols);
}

bool MaslovaURowMatrVecMultSEQ::PreProcessingImpl() {
  const auto &matrix = GetInput().first;
  if (!matrix.data.empty() && matrix.rows > 0) {
    GetOutput().assign(matrix.rows, 0.0);
  }
  return true;
}

bool MaslovaURowMatrVecMultSEQ::RunImpl() {
  const auto &matrix = GetInput().first;
  const auto &vec = GetInput().second;

  if (matrix.data.empty() || matrix.rows == 0 || matrix.cols == 0) {
    return true;
  }

  auto &res = GetOutput();

  if (res.size() != matrix.rows) {
    res.assign(matrix.rows, 0.0);
  }

  for (size_t i = 0; i < matrix.rows; ++i) {
    double sum = 0.0;
    const size_t offset = i * matrix.cols;
    for (size_t j = 0; j < matrix.cols; ++j) {
      sum += matrix.data[offset + j] * vec[j];
    }
    res[i] = sum;
  }

  return true;
}

bool MaslovaURowMatrVecMultSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace maslova_u_row_matr_vec_mult
