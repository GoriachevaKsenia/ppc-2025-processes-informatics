#include "khruev_a_gauss_jordan/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "khruev_a_gauss_jordan/common/include/common.hpp"

namespace khruev_a_gauss_jordan {

namespace {

constexpr double kTol = 1e-10;

bool NearZero(double x) {
  return std::fabs(x) < kTol;
}

int SelectPivot(const std::vector<std::vector<double>> &a, int start_row, int col) {
  int best = -1;
  double max_val = 0.0;

  for (int i = start_row; std::cmp_less(i, static_cast<int>(a.size())); ++i) {
    double v = std::fabs(a[i][col]);
    if (v > max_val) {
      max_val = v;
      best = i;
    }
  }
  return (max_val > kTol) ? best : -1;
}

void Normalize(std::vector<double> &row, int pivot_col) {
  double p = row[pivot_col];
  if (NearZero(p)) {
    return;
  }
  for (double &x : row) {
    x /= p;
  }
}

void Eliminate(std::vector<double> &row, const std::vector<double> &pivot, double factor) {
  for (size_t j = 0; j < row.size(); ++j) {
    row[j] -= factor * pivot[j];
  }
}

bool RowIsZero(const std::vector<double> &row, int until) {
  for (int j = 0; j < until; ++j) {
    if (!NearZero(row[j])) {
      return false;
    }
  }
  return true;
}

}  // namespace

KhruevAGaussJordanSEQ::KhruevAGaussJordanSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool KhruevAGaussJordanSEQ::ValidationImpl() {
  if (GetInput().empty()) {
    return true;
  }

  size_t cols = GetInput()[0].size();
  for (const auto &r : GetInput()) {
    if (r.size() != cols) {
      return false;
    }
  }

  return GetOutput().empty() && cols > 0;
}

bool KhruevAGaussJordanSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

void KhruevAGaussJordanSEQ::ToReducedForm(std::vector<std::vector<double>> &a) {
  int n = static_cast<int>(a.size());
  int m = static_cast<int>(a[0].size());

  int row = 0;
  for (int col = 0; col < m - 1 && row < n; ++col) {
    int pivot = 0;
    pivot = SelectPivot(a, row, col);
    if (pivot == -1) {
      continue;
    }

    std::swap(a[row], a[pivot]);
    Normalize(a[row], col);

    for (int i = 0; i < n; ++i) {
      if (i == row) {
        continue;
      }
      double factor = a[i][col];
      if (!NearZero(factor)) {
        Eliminate(a[i], a[row], factor);
      }
    }
    ++row;
  }
}

bool KhruevAGaussJordanSEQ::DetectInconsistency(const std::vector<std::vector<double>> &a) {
  const int m = static_cast<int>(a[0].size());

  return std::ranges::any_of(
      a, [m](const std::vector<double> &row) { return RowIsZero(row, m - 1) && !NearZero(row[m - 1]); });
}

int KhruevAGaussJordanSEQ::ComputeRank(const std::vector<std::vector<double>> &a) {
  int rank = 0;
  int m = static_cast<int>(a[0].size());
  for (const auto &row : a) {
    if (!RowIsZero(row, m - 1)) {
      ++rank;
    }
  }
  return rank;
}

std::vector<double> KhruevAGaussJordanSEQ::RecoverSolution(const std::vector<std::vector<double>> &a) {
  int n = static_cast<int>(a.size());
  int m = static_cast<int>(a[0].size());

  std::vector<double> sol(m - 1, 0.0);
  for (int i = 0; i < std::min(n, m - 1); ++i) {
    for (int j = 0; j < m - 1; ++j) {
      if (!NearZero(a[i][j])) {
        sol[j] = a[i][m - 1];
        break;
      }
    }
  }
  return sol;
}

bool KhruevAGaussJordanSEQ::RunImpl() {
  if (GetInput().empty()) {
    return true;
  }
  auto a = GetInput();
  ToReducedForm(a);

  if (DetectInconsistency(a)) {
    GetOutput().clear();
    return false;
  }

  int rank = 0;
  rank = ComputeRank(a);
  if (rank < static_cast<int>(a[0].size()) - 1 && std::cmp_less(rank, static_cast<int>(a.size()))) {
    GetOutput().clear();
    return false;
  }

  GetOutput() = RecoverSolution(a);
  return true;
}

bool KhruevAGaussJordanSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace khruev_a_gauss_jordan
