#include "eremin_v_strongin_algorithm/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <tuple>
#include <vector>

#include "eremin_v_strongin_algorithm/common/include/common.hpp"

namespace eremin_v_strongin_algorithm {

EreminVStronginAlgorithmSEQ::EreminVStronginAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool EreminVStronginAlgorithmSEQ::ValidationImpl() {
  auto &input = GetInput();

  double lower_bound = std::get<0>(input);
  double upper_bound = std::get<1>(input);
  double epsilon = std::get<2>(input);
  int max_iters = std::get<3>(input);

  return (lower_bound < upper_bound) && (epsilon > 0.0 && epsilon <= (upper_bound - lower_bound)) &&
         (max_iters > 0 && max_iters <= 100000000) && (lower_bound >= -1e9 && lower_bound <= 1e9) &&
         (upper_bound >= -1e9 && upper_bound <= 1e9) && (GetOutput() == 0);
}

bool EreminVStronginAlgorithmSEQ::PreProcessingImpl() {
  return true;
}

double EreminVStronginAlgorithmSEQ::CalculateLipschitzEstimate(const std::vector<double> &search_points,
                                                               const std::vector<double> &function_values) {
  double lipschitz_estimate = 0.0;
  for (std::size_t i = 1; i < search_points.size(); ++i) {
    double interval_width = search_points[i] - search_points[i - 1];
    double value_difference = std::abs(function_values[i] - function_values[i - 1]);
    double current_slope = value_difference / interval_width;
    lipschitz_estimate = std::max(current_slope, lipschitz_estimate);
  }
  return lipschitz_estimate;
}

EreminVStronginAlgorithmSEQ::IntervalCharacteristic EreminVStronginAlgorithmSEQ::FindBestInterval(
    const std::vector<double> &search_points, const std::vector<double> &function_values, double m_parameter) {
  IntervalCharacteristic best{.value = -1e18, .index = 1};

  for (std::size_t i = 1; i < search_points.size(); ++i) {
    double interval_width = search_points[i] - search_points[i - 1];
    double characteristic = (m_parameter * interval_width) +
                            ((function_values[i] - function_values[i - 1]) *
                             (function_values[i] - function_values[i - 1]) / (m_parameter * interval_width)) -
                            (2.0 * (function_values[i] + function_values[i - 1]));

    if (characteristic > best.value) {
      best.value = characteristic;
      best.index = static_cast<int>(i);
    }
  }

  return best;
}

bool EreminVStronginAlgorithmSEQ::RunImpl() {
  auto &input = GetInput();
  double lower_bound = std::get<0>(input);
  double upper_bound = std::get<1>(input);
  double epsilon = std::get<2>(input);
  int max_iterations = std::get<3>(input);

  std::function<double(double)> objective_function = std::get<4>(GetInput());
  ;

  std::vector<double> search_points = {lower_bound, upper_bound};
  std::vector<double> function_values = {objective_function(lower_bound), objective_function(upper_bound)};
  search_points.reserve(max_iterations + 2);
  function_values.reserve(max_iterations + 2);

  int current_iteration = 0;
  double r_coefficient = 2.0;
  double max_interval_width = upper_bound - lower_bound;

  while (max_interval_width > epsilon && current_iteration < max_iterations) {
    ++current_iteration;

    double lipschitz_estimate = CalculateLipschitzEstimate(search_points, function_values);

    double m_parameter = (lipschitz_estimate > 0.0) ? r_coefficient * lipschitz_estimate : 1.0;

    auto best_interval = FindBestInterval(search_points, function_values, m_parameter);
    std::size_t best_interval_index = best_interval.index;

    double left_point = search_points[best_interval_index - 1];
    double right_point = search_points[best_interval_index];
    double left_value = function_values[best_interval_index - 1];
    double right_value = function_values[best_interval_index];

    double new_point = (0.5 * (left_point + right_point)) - ((right_value - left_value) / (2.0 * m_parameter));

    if (new_point <= left_point || new_point >= right_point) {
      new_point = 0.5 * (left_point + right_point);
    }

    double new_value = objective_function(new_point);

    search_points.insert(search_points.begin() + static_cast<std::ptrdiff_t>(best_interval_index), new_point);
    function_values.insert(function_values.begin() + static_cast<std::ptrdiff_t>(best_interval_index), new_value);

    max_interval_width = 0.0;
    for (std::size_t i = 1; i < search_points.size(); ++i) {
      double current_width = search_points[i] - search_points[i - 1];
      max_interval_width = std::max(current_width, max_interval_width);
    }
  }

  GetOutput() = *std::ranges::min_element(function_values);
  return true;
}

bool EreminVStronginAlgorithmSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace eremin_v_strongin_algorithm
