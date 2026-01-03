#include "sannikov_i_horizontal_band_gauss/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "sannikov_i_horizontal_band_gauss/common/include/common.hpp"

namespace sannikov_i_horizontal_band_gauss {

SannikovIHorizontalBandGaussSEQ::SannikovIHorizontalBandGaussSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &input_buffer = GetInput();
  InType tmp(in);
  input_buffer.swap(tmp);
  GetOutput().clear();
}

bool SannikovIHorizontalBandGaussSEQ::ValidationImpl() {
  (void)this;

  const auto &a = std::get<0>(GetInput());
  const auto &rhs = std::get<1>(GetInput());

  if (a.empty() || rhs.empty()) {
    return false;
  }
  if (a.size() != rhs.size()) {
    return false;
  }
  return GetOutput().empty();
}

bool SannikovIHorizontalBandGaussSEQ::PreProcessingImpl() {
  (void)this;

  GetOutput().clear();
  return GetOutput().empty();
}

bool SannikovIHorizontalBandGaussSEQ::RunImpl() {
  (void)this;

  auto a = std::get<0>(GetInput());
  auto rhs = std::get<1>(GetInput());
  const std::size_t band = std::get<2>(GetInput());

  const std::size_t n = a.size();
  GetOutput().assign(n, 0.0);

  constexpr double kEps = 1e-8;

  for (std::size_t i = 0; i < n; ++i) {
    const double pivot = a[i][i];
    if (std::fabs(pivot) < kEps) {
      return false;
    }

    const std::size_t j_end = std::min(n, i + band + 1);

    for (std::size_t j = i; j < j_end; ++j) {
      a[i][j] /= pivot;
    }
    rhs[i] /= pivot;

    const std::size_t row_end = std::min(n, i + band + 1);
    for (std::size_t row = i + 1; row < row_end; ++row) {
      const double mult = a[row][i];

      for (std::size_t j = i; j < j_end; ++j) {
        a[row][j] -= mult * a[i][j];
      }
      rhs[row] -= mult * rhs[i];
    }
  }

  for (std::size_t idx = 0; idx < n; ++idx) {
    const std::size_t i = n - 1 - idx;
    double sum = rhs[i];

    const std::size_t j_end = std::min(n, i + band + 1);
    for (std::size_t j = i + 1; j < j_end; ++j) {
      sum -= a[i][j] * GetOutput()[j];
    }
    GetOutput()[i] = sum;
  }

  return !GetOutput().empty();
}

bool SannikovIHorizontalBandGaussSEQ::PostProcessingImpl() {
  (void)this;

  return !GetOutput().empty();
}

}  // namespace sannikov_i_horizontal_band_gauss
