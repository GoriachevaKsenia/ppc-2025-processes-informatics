#include "kruglova_a_vertical_ribbon_matvec/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "kruglova_a_vertical_ribbon_matvec/common/include/common.hpp"

namespace kruglova_a_vertical_ribbon_matvec {

KruglovaAVerticalRibbMatSEQ::KruglovaAVerticalRibbMatSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool KruglovaAVerticalRibbMatSEQ::ValidationImpl() {
  const int rows = std::get<0>(GetInput());
  const int cols = std::get<1>(GetInput());
  const std::vector<double> &matrix = std::get<2>(GetInput());
  const std::vector<double> &vec = std::get<3>(GetInput());

  if (rows <= 0 || cols <= 0) {
    return false;
  }

  if (matrix.size() != static_cast<size_t>(rows) * static_cast<size_t>(cols)) {
    return false;
  }

  if (vec.size() != static_cast<size_t>(cols)) {
    return false;
  }
  return true;
}

bool KruglovaAVerticalRibbMatSEQ::PreProcessingImpl() {
  const int rows = std::get<0>(GetInput());
  GetOutput().assign(rows, 0.0);
  return true;
}

bool KruglovaAVerticalRibbMatSEQ::RunImpl() {
  const int rows = std::get<0>(GetInput());
  const int cols = std::get<1>(GetInput());
  const std::vector<double> &matrix = std::get<2>(GetInput());
  const std::vector<double> &vec_b = std::get<3>(GetInput());

  std::vector<double> &res_y = GetOutput();

  for (int i = 0; i < rows; ++i) {
    double sum = 0.0;
    for (int j = 0; j < cols; ++j) {
      sum += matrix[(i * cols) + j] * vec_b[j];
    }
    res_y[i] = sum;
  }

  return true;
}

bool KruglovaAVerticalRibbMatSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kruglova_a_vertical_ribbon_matvec
