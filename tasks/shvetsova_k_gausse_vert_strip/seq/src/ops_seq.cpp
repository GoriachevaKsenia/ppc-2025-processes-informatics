#include "shvetsova_k_gausse_vert_strip/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "shvetsova_k_gausse_vert_strip/common/include/common.hpp"

namespace shvetsova_k_gausse_vert_strip {

ShvetsovaKGaussVertStripSEQ::ShvetsovaKGaussVertStripSEQ(const InType &in) : size_of_rib_(1) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &input_buffer = GetInput();
  InType tmp(in);
  input_buffer.swap(tmp);
  GetOutput().clear();
}

bool ShvetsovaKGaussVertStripSEQ::ValidationImpl() {
  const auto &matrix = GetInput().first;
  const auto &vec = GetInput().second;
  if (matrix.empty()) {
    return true;
  }
  return matrix.size() == vec.size();
}

bool ShvetsovaKGaussVertStripSEQ::PreProcessingImpl() {
  const auto &matrix = GetInput().first;
  int n = static_cast<int>(matrix.size());
  if (n == 0) {
    return true;
  }

  size_of_rib_ = 1;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i != j && std::abs(matrix[i][j]) > 1e-12) {
        size_of_rib_ = std::max(size_of_rib_, std::abs(i - j) + 1);
      }
    }
  }
  return true;
}

void ShvetsovaKGaussVertStripSEQ::ProcessRow(int i, int row_end, std::vector<std::vector<double>> &a,
                                             std::vector<double> &x, double eps) {
  double pivot = a[i][i];
  if (std::abs(pivot) < eps) {
    pivot = (pivot >= 0) ? eps : -eps;
  }

  for (int j = i; j < row_end; ++j) {
    a[i][j] /= pivot;
  }
  x[i] /= pivot;

  for (int k = i + 1; k < row_end; ++k) {
    double factor = a[k][i];
    if (std::abs(factor) < eps) {
      continue;
    }
    for (int j = i; j < row_end; ++j) {
      a[k][j] -= factor * a[i][j];
    }
    x[k] -= factor * x[i];
  }
}

void ShvetsovaKGaussVertStripSEQ::ForwardElimination(int n, std::vector<std::vector<double>> &a,
                                                     std::vector<double> &x) const {
  const double eps = 1e-15;
  for (int i = 0; i < n; ++i) {
    int row_end = std::min(n, i + size_of_rib_);
    ProcessRow(i, row_end, a, x, eps);
  }
}

std::vector<double> ShvetsovaKGaussVertStripSEQ::BackSubstitution(int n, const std::vector<std::vector<double>> &a,
                                                                  const std::vector<double> &x) const {
  std::vector<double> res(n);
  for (int i = n - 1; i >= 0; --i) {
    double sum = x[i];
    int row_end = std::min(n, i + size_of_rib_);
    for (int j = i + 1; j < row_end; ++j) {
      sum -= a[i][j] * res[j];
    }
    res[i] = sum;
  }
  return res;
}

bool ShvetsovaKGaussVertStripSEQ::RunImpl() {
  const auto &matrix_a = GetInput().first;
  const auto &vec_b = GetInput().second;
  int n = static_cast<int>(matrix_a.size());

  if (n == 0) {
    return true;
  }

  std::vector<std::vector<double>> a = matrix_a;
  std::vector<double> x = vec_b;

  ForwardElimination(n, a, x);
  GetOutput() = BackSubstitution(n, a, x);

  return true;
}

bool ShvetsovaKGaussVertStripSEQ::PostProcessingImpl() {
  if (GetInput().first.empty()) {
    return true;
  }
  return !GetOutput().empty();
}

}  // namespace shvetsova_k_gausse_vert_strip
