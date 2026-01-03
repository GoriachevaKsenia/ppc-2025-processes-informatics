#include "goriacheva_k_strassen_algorithm/seq/include/ops_seq.hpp"

#include "goriacheva_k_strassen_algorithm/common/include/common.hpp"

namespace goriacheva_k_strassen_algorithm {

GoriachevaKStrassenAlgorithmSEQ::GoriachevaKStrassenAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool GoriachevaKStrassenAlgorithmSEQ::ValidationImpl() {
  return IsSquare(GetInput().a) && IsSquare(GetInput().b) && GetInput().a.size() == GetInput().a.size();
}

bool GoriachevaKStrassenAlgorithmSEQ::PreProcessingImpl() {
  input_matrices_ = GetInput();
  return true;
}

bool GoriachevaKStrassenAlgorithmSEQ::RunImpl() {
  const auto &a = input_matrices_.a;
  const auto &b = input_matrices_.b;

  std::size_t n = a.size();
  std::size_t m = NextPowerOfTwo(n);

  Matrix a_pad = (n == m) ? a : PadMatrix(a, m);
  Matrix b_pad = (n == m) ? b : PadMatrix(b, m);

  Matrix c_pad = Strassen(a_pad, b_pad);
  result_matrix_ = (n == m) ? c_pad : CropMatrix(c_pad, n);

  return true;
}

bool GoriachevaKStrassenAlgorithmSEQ::PostProcessingImpl() {
  GetOutput() = result_matrix_;
  return true;
}

}  // namespace goriacheva_k_strassen_algorithm
