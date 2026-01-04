#include "liulin_y_integ_mnog_func_monte_carlo/seq/include/ops_seq.hpp"

#include <cstdint>
#include <random>

#include "liulin_y_integ_mnog_func_monte_carlo/common/include/common.hpp"

namespace liulin_y_integ_mnog_func_monte_carlo {

// Remove the thread_local static std::mt19937 gen declaration since it's not used

LiulinYIntegMnogFuncMonteCarloSEQ::LiulinYIntegMnogFuncMonteCarloSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool LiulinYIntegMnogFuncMonteCarloSEQ::ValidationImpl() {
  const auto &input = GetInput();
  return input.num_points > 0 && input.x_min <= input.x_max && input.y_min <= input.y_max;
}

bool LiulinYIntegMnogFuncMonteCarloSEQ::PreProcessingImpl() {
  return true;
}

bool LiulinYIntegMnogFuncMonteCarloSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &result = GetOutput();

  if (input.num_points <= 0) {
    result = 0.0;
    return true;
  }

  const double area = (input.x_max - input.x_min) * (input.y_max - input.y_min);
  if (area <= 0.0) {
    result = 0.0;
    return true;
  }

  const double x_min = input.x_min;
  const double x_range = input.x_max - x_min;
  const double y_min = input.y_min;
  const double y_range = input.y_max - y_min;
  const auto &func = input.f;
  const int64_t n = input.num_points;

  static thread_local std::minstd_rand fast_gen(std::random_device{}());

  std::uniform_real_distribution<double> dist_x(0.0, x_range);
  std::uniform_real_distribution<double> dist_y(0.0, y_range);

  double sum = 0.0;
  int64_t i = 0;

  // Пришлось развернуть цикл вручную для оптимизации производительности
  for (; i + 3 < n; i += 4) {
    const double x1 = x_min + dist_x(fast_gen);
    const double y1 = y_min + dist_y(fast_gen);
    const double x2 = x_min + dist_x(fast_gen);
    const double y2 = y_min + dist_y(fast_gen);
    const double x3 = x_min + dist_x(fast_gen);
    const double y3 = y_min + dist_y(fast_gen);
    const double x4 = x_min + dist_x(fast_gen);
    const double y4 = y_min + dist_y(fast_gen);

    sum += func(x1, y1) + func(x2, y2) + func(x3, y3) + func(x4, y4);
  }

  for (; i < n; ++i) {
    const double x = x_min + dist_x(fast_gen);
    const double y = y_min + dist_y(fast_gen);
    sum += func(x, y);
  }

  result = (sum * area) / static_cast<double>(n);

  return true;
}

bool LiulinYIntegMnogFuncMonteCarloSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace liulin_y_integ_mnog_func_monte_carlo
