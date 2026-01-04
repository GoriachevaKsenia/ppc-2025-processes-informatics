#include "frolova_s_sum_elem_matrix/seq/include/ops_seq.hpp"

#include <cmath>
// #include <cstddef>
#include <tuple>
#include <vector>

#include "frolova_s_sum_elem_matrix/common/include/common.hpp"

namespace frolova_s_sum_elem_matrix {

FrolovaSSumElemMatrixSEQ::FrolovaSSumElemMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool FrolovaSSumElemMatrixSEQ::ValidationImpl() {
  const auto &in = GetInput();
  const auto &my_matrix = std::get<0>(in);
  int param_dim1 = std::get<1>(in);
  int param_dim2 = std::get<2>(in);

  return (param_dim1 > 0 && param_dim2 > 0 && static_cast<int>(my_matrix.size()) == (param_dim1 * param_dim2));
}

bool FrolovaSSumElemMatrixSEQ::PreProcessingImpl() {
  return true;
}

bool FrolovaSSumElemMatrixSEQ::RunImpl() {
  const std::vector<double> &vect_data = std::get<0>(GetInput());

  double all_sum = 0;
  for (double val : vect_data) {
    all_sum += val;
  }

  GetOutput() = all_sum;
  return true;
}

bool FrolovaSSumElemMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace frolova_s_sum_elem_matrix
