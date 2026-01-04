#include "kosolapov_v_gauss_method_tape_hor_scheme/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "kosolapov_v_gauss_method_tape_hor_scheme/common/include/common.hpp"

namespace kosolapov_v_gauss_method_tape_hor_scheme {

KosolapovVGaussMethodTapeHorSchemeSEQ::KosolapovVGaussMethodTapeHorSchemeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = {};
}

bool KosolapovVGaussMethodTapeHorSchemeSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.matrix.empty()) {
    return false;
  }
  int n = static_cast<int>(input.matrix.size());
  if (input.matrix.size() != static_cast<size_t>(n)) {
    return false;
  }
  for (const auto &row : input.matrix) {
    if (row.size() != static_cast<size_t>(n)) {
      return false;
    }
  }
  return input.r_side.size() == static_cast<size_t>(n);
}

bool KosolapovVGaussMethodTapeHorSchemeSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  int n = static_cast<int>(input.matrix.size());
  GetOutput() = std::vector<double>(n, 0.0);
  return true;
}

bool KosolapovVGaussMethodTapeHorSchemeSEQ::RunImpl() {
  const auto &input = GetInput();
  int n = static_cast<int>(input.matrix.size());

  std::vector<std::vector<double>> a = input.matrix;
  std::vector<double> b = input.r_side;
  std::vector<int> col_order(n);
  for (int i = 0; i < n; i++) {
    col_order[i] = i;
  }
  ForwardElimination(a, b, col_order, n);
  std::vector<double> result = BackwardSubstitution(a, b, col_order, n);
  GetOutput() = result;
  return true;
}

bool KosolapovVGaussMethodTapeHorSchemeSEQ::PostProcessingImpl() {
  return true;
}

void KosolapovVGaussMethodTapeHorSchemeSEQ::ForwardElimination(std::vector<std::vector<double>> &a,
                                                               std::vector<double> &b, std::vector<int> &col_order,
                                                               int n) {
  for (int i = 0; i < n; i++) {
    int leading_col = i;
    double max_elem = std::abs(a[i][i]);
    SelectPivot(i, n, a, max_elem, leading_col);
    SwapRows(leading_col, n, i, a, col_order);
    double cur_el = a[i][i];
    for (int j = i; j < n; j++) {
      a[i][j] /= cur_el;
    }
    b[i] /= cur_el;
    RowSub(i, n, a, b);
  }
}
std::vector<double> KosolapovVGaussMethodTapeHorSchemeSEQ::BackwardSubstitution(std::vector<std::vector<double>> &a,
                                                                                std::vector<double> &b,
                                                                                std::vector<int> &col_order, int n) {
  std::vector<double> output(n);
  for (int i = n - 1; i >= 0; --i) {
    output[col_order[i]] = b[i];
    for (int j = i + 1; j < n; ++j) {
      output[col_order[i]] -= a[i][j] * output[col_order[j]];
    }
  }
  return output;
}
void KosolapovVGaussMethodTapeHorSchemeSEQ::SelectPivot(int i, int n, const std::vector<std::vector<double>> &a,
                                                        double &max_elem, int &leading_col) {
  for (int j = i + 1; j < n; j++) {
    if (std::abs(a[i][j]) > max_elem) {
      max_elem = std::abs(a[i][j]);
      leading_col = j;
    }
  }
}
void KosolapovVGaussMethodTapeHorSchemeSEQ::SwapRows(int leading_col, int n, int i, std::vector<std::vector<double>> &a,
                                                     std::vector<int> &col_order) {
  if (leading_col != i) {
    for (int k = 0; k < n; k++) {
      std::swap(a[k][i], a[k][leading_col]);
    }
    std::swap(col_order[i], col_order[leading_col]);
  }
}
void KosolapovVGaussMethodTapeHorSchemeSEQ::RowSub(int i, int n, std::vector<std::vector<double>> &a,
                                                   std::vector<double> &b) {
  for (int k = i + 1; k < n; k++) {
    double ratio = a[k][i];
    for (int j = i; j < n; j++) {
      a[k][j] -= ratio * a[i][j];
    }
    b[k] -= ratio * b[i];
  }
}
}  // namespace kosolapov_v_gauss_method_tape_hor_scheme
