#include "pylaeva_s_simple_iteration_method/seq/include/ops_seq.hpp"

#include <cctype>
#include <cmath>
#include <cstddef>
#include <vector>

#include "pylaeva_s_simple_iteration_method/common/include/common.hpp"

namespace pylaeva_s_simple_iteration_method {
namespace {

constexpr double kEps = 1e-6;
constexpr int kMaxIterations = 10000;

bool DiagonalDominance(const std::vector<double> &a, size_t n) {
  for (size_t i = 0; i < n; i++) {
    double diag = std::fabs(a[(i * n) + i]);  // Модуль диагонального элемента
    double row_sum = 0.0;                     // Сумма модулей недиагональных элементов строки

    for (size_t j = 0; j < n; j++) {
      if (j != i) {
        row_sum += std::fabs(a[(i * n) + j]);
      }
    }
    // Проверка строгого диагонального преобладания:
    // Диагональный элемент должен быть БОЛЬШЕ суммы остальных элементов строки
    if (diag <= row_sum) {
      return false;
    }
  }
  return true;
}
}  // namespace

PylaevaSSimpleIterationMethodSEQ::PylaevaSSimpleIterationMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool PylaevaSSimpleIterationMethodSEQ::ValidationImpl() {
  const auto &n = std::get<0>(GetInput());
  const auto &a = std::get<1>(GetInput());
  const auto &b = std::get<2>(GetInput());
  return ((n > 0) && (a.size() == n * n) && (b.size() == n) && (DiagonalDominance(a, n)));
}

bool PylaevaSSimpleIterationMethodSEQ::PreProcessingImpl() {
  return true;
}

bool PylaevaSSimpleIterationMethodSEQ::RunImpl() {
  const auto &n = std::get<0>(GetInput());
  const auto &a = std::get<1>(GetInput());
  const auto &b = std::get<2>(GetInput());

  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);

  for (int iter = 0; iter < kMaxIterations; ++iter) {
    for (size_t i = 0; i < n; ++i) {
      double sum = 0.0;
      for (size_t j = 0; j < n; ++j) {
        if (j != i) {
          sum += a[(i * n) + j] * x[j];
        }
      }
      x_new[i] = (b[i] - sum) / a[(i * n) + i];
    }

    double norm = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double diff = x_new[i] - x[i];
      norm += diff * diff;
    }

    x = x_new;

    if (std::sqrt(norm) < kEps) {
      break;
    }
  }

  GetOutput() = x;
  return true;
}

bool PylaevaSSimpleIterationMethodSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace pylaeva_s_simple_iteration_method
