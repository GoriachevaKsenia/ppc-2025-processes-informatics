#include "romanova_v_jacobi_method/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "romanova_v_jacobi_method/common/include/common.hpp"

namespace romanova_v_jacobi_method {

RomanovaVJacobiMethodSEQ::RomanovaVJacobiMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput();
}

bool RomanovaVJacobiMethodSEQ::ValidationImpl() {
  bool status = true;
  // матрица имеет корректные размеры
  std::vector<std::vector<double>> a = std::get<1>(GetInput());
  std::vector<double> x = std::get<0>(GetInput());
  std::vector<double> b = std::get<2>(GetInput());
  status = status && !a.empty();
  for (size_t i = 0; i < a.size(); i++) {
    status = status && (a.size() == a[i].size());
  }

  status = status && IsDiagonallyDominant(a);

  status = status && (a.size() == x.size());
  status = status && (a.size() == b.size());
  return status;
}

bool RomanovaVJacobiMethodSEQ::PreProcessingImpl() {
  std::tie(x_, A_, b_, eps_, maxIterations_) = GetInput();
  // x_ = std::get<0>(GetInput());
  // A_ = std::get<1>(GetInput());
  // b_ = std::get<2>(GetInput());
  // eps_ = std::get<3>(GetInput());
  // maxIterations_ = std::get<4>(GetInput());
  size_ = x_.size();
  return true;
}

bool RomanovaVJacobiMethodSEQ::RunImpl() {
  size_t k = 0;
  std::vector<double> prev(x_.size(), 0.0);
  double diff = eps_;
  while (diff >= eps_ && k < maxIterations_) {
    diff = 0.0;
    prev = x_;
    for (size_t i = 0; i < size_; i++) {
      double sum = 0.0;
      for (size_t j = 0; j < size_; j++) {
        if (i != j) {
          sum += A_[i][j] * prev[j];
        }
      }
      x_[i] = (b_[i] - sum) / A_[i][i];
      diff = std::max(diff, std::abs(x_[i] - prev[i]));
    }
    k++;
  }

  return true;
}

bool RomanovaVJacobiMethodSEQ::PostProcessingImpl() {
  GetOutput() = x_;
  return true;
}

bool RomanovaVJacobiMethodSEQ::IsDiagonallyDominant(const std::vector<std::vector<double>> &matrix) {
  for (size_t i = 0; i < matrix.size(); i++) {
    double sum = 0.0;
    for (size_t j = 0; j < matrix[i].size(); j++) {
      if (i != j) {
        sum += matrix[i][j];
      }
    }
    if (matrix[i][i] <= sum) {
      return false;
    }
  }
  return true;
}

}  // namespace romanova_v_jacobi_method
