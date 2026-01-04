#include "spichek_d_jacobi/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "spichek_d_jacobi/common/include/common.hpp"

namespace spichek_d_jacobi {

SpichekDJacobiSEQ::SpichekDJacobiSEQ(InType in) : input_(std::move(in)) {
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool SpichekDJacobiSEQ::ValidationImpl() {
  const auto &[A, b, eps, max_iter] = input_;
  return !A.empty() && A.size() == b.size() && eps > 0 && max_iter > 0;
}

bool SpichekDJacobiSEQ::PreProcessingImpl() {
  GetInput() = input_;
  GetOutput().assign(std::get<1>(input_).size(), 0.0);
  return true;
}

bool SpichekDJacobiSEQ::RunImpl() {
  const auto &[A, b, eps, max_iter] = GetInput();
  size_t n = b.size();

  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);

  for (int iter = 0; iter < max_iter; ++iter) {
    double max_diff = 0.0;

    for (size_t i = 0; i < n; ++i) {
      double sum = 0.0;
      for (size_t j = 0; j < n; ++j) {
        if (j != i) {
          sum += A[i][j] * x[j];
        }
      }
      x_new[i] = (b[i] - sum) / A[i][i];
      max_diff = std::max(max_diff, std::abs(x_new[i] - x[i]));
    }

    x = x_new;
    if (max_diff < eps) {
      break;
    }
  }

  GetOutput() = x;
  return true;
}

bool SpichekDJacobiSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_jacobi
