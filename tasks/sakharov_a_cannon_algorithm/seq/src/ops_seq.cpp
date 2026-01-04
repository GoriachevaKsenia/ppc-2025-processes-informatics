#include "sakharov_a_cannon_algorithm/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "sakharov_a_cannon_algorithm/common/include/common.hpp"

namespace sakharov_a_cannon_algorithm {

SakharovACannonAlgorithmSEQ::SakharovACannonAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SakharovACannonAlgorithmSEQ::ValidationImpl() {
  return IsValidInput(GetInput());
}

bool SakharovACannonAlgorithmSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  auto out_size = static_cast<std::size_t>(input.rows_a) * static_cast<std::size_t>(input.cols_b);
  GetOutput().assign(out_size, 0.0);
  return true;
}

bool SakharovACannonAlgorithmSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  const int m = input.rows_a;
  const int k = input.cols_a;
  const int n = input.cols_b;

  for (int ii = 0; ii < m; ++ii) {
    for (int kk = 0; kk < k; ++kk) {
      double a_val = input.a[Idx(k, ii, kk)];
      for (int jj = 0; jj < n; ++jj) {
        output[Idx(n, ii, jj)] += a_val * input.b[Idx(n, kk, jj)];
      }
    }
  }

  return true;
}

bool SakharovACannonAlgorithmSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sakharov_a_cannon_algorithm
