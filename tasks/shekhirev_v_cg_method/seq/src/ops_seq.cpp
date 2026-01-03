#include "../include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "../../common/include/common.hpp"

namespace shekhirev_v_cg_method_seq {

ConjugateGradientSeq::ConjugateGradientSeq(const shekhirev_v_cg_method::InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ConjugateGradientSeq::ValidationImpl() {
  const int n = GetInput().n;
  return n > 0 && GetInput().A.size() == static_cast<size_t>(n) * n && GetInput().b.size() == static_cast<size_t>(n);
}

bool ConjugateGradientSeq::PreProcessingImpl() {
  return true;
}

std::vector<double> ConjugateGradientSeq::MultiplyMatrixVector(const std::vector<double> &matrix,
                                                               const std::vector<double> &vec, int n) {
  std::vector<double> result(n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      size_t index = (static_cast<size_t>(i) * n) + j;
      result[i] += matrix[index] * vec[j];
    }
  }
  return result;
}

double ConjugateGradientSeq::DotProduct(const std::vector<double> &a, const std::vector<double> &b) {
  double res = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    res += (a[i] * b[i]);
  }
  return res;
}

bool ConjugateGradientSeq::RunImpl() {
  const auto &a_mat = GetInput().A;
  const auto &b_vec = GetInput().b;
  const int n = GetInput().n;

  std::vector<double> x(n, 0.0);
  std::vector<double> r = b_vec;
  std::vector<double> p = r;

  double rs_old = DotProduct(r, r);

  const int max_iter = n * 2;
  const double epsilon = 1e-10;

  for (int k = 0; k < max_iter; ++k) {
    std::vector<double> ap = MultiplyMatrixVector(a_mat, p, n);
    double p_ap = DotProduct(p, ap);

    const double alpha = rs_old / p_ap;

    for (int i = 0; i < n; ++i) {
      x[i] += (alpha * p[i]);
      r[i] -= (alpha * ap[i]);
    }

    double rs_new = DotProduct(r, r);

    if (std::sqrt(rs_new) < epsilon) {
      break;
    }

    const double beta = rs_new / rs_old;

    for (int i = 0; i < n; ++i) {
      p[i] = r[i] + (beta * p[i]);
    }

    rs_old = rs_new;
  }

  GetOutput() = x;
  return true;
}

bool ConjugateGradientSeq::PostProcessingImpl() {
  return true;
}

}  // namespace shekhirev_v_cg_method_seq
