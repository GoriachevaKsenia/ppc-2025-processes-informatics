#include "kiselev_i_gauss_method_horizontal_tape_scheme/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "kiselev_i_gauss_method_horizontal_tape_scheme/common/include/common.hpp"

namespace kiselev_i_gauss_method_horizontal_tape_scheme {

KiselevITestTaskSEQ::KiselevITestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &buf = GetInput();
  InType tmp(in);
  buf.swap(tmp);
  GetOutput().clear();
}

bool KiselevITestTaskSEQ::ValidationImpl() {
  const auto &a_vector = std::get<0>(GetInput());
  const auto &b_vector = std::get<1>(GetInput());

  return !a_vector.empty() && a_vector.size() == b_vector.size() && GetOutput().empty();
}

bool KiselevITestTaskSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

namespace {

void BackSubstitution(const std::vector<std::vector<double>> &mat, const std::vector<double> &rhs,
                      std::vector<double> &result, std::size_t band) {
  const std::size_t num = mat.size();

  for (std::size_t step = 0; step < num; ++step) {
    const std::size_t index = num - 1 - step;
    double acc = rhs[index];

    const std::size_t col_end = std::min(num, index + band + 1);
    for (std::size_t j = index + 1; j < col_end; ++j) {
      acc -= mat[index][j] * result[j];
    }

    result[index] = acc;
  }
}

}  // namespace

bool KiselevITestTaskSEQ::RunImpl() {
  auto mat = std::get<0>(GetInput());
  auto rhs = std::get<1>(GetInput());
  const std::size_t band = std::get<2>(GetInput());

  const std::size_t num = mat.size();
  GetOutput().assign(num, 0.0);

  constexpr double kEps = 1e-12;

  for (std::size_t k_index = 0; k_index < num; ++k_index) {
    const double diag = mat[k_index][k_index];
    if (std::fabs(diag) < kEps) {
      return false;
    }

    const std::size_t col_end = std::min(num, k_index + band + 1);

    for (std::size_t j_index = k_index; j_index < col_end; ++j_index) {
      mat[k_index][j_index] /= diag;
    }
    rhs[k_index] /= diag;

    const std::size_t row_end = std::min(num, k_index + band + 1);
    for (std::size_t index = k_index + 1; index < row_end; ++index) {
      const double factor = mat[index][k_index];
      if (factor == 0.0) {
        continue;
      }

      for (std::size_t j_index = k_index; j_index < col_end; ++j_index) {
        mat[index][j_index] -= factor * mat[k_index][j_index];
      }
      rhs[index] -= factor * rhs[k_index];
    }
  }

  BackSubstitution(mat, rhs, GetOutput(), band);
  return true;
}

bool KiselevITestTaskSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace kiselev_i_gauss_method_horizontal_tape_scheme
