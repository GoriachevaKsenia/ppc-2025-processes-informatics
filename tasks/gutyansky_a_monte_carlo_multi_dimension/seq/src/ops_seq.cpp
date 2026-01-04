#include "gutyansky_a_monte_carlo_multi_dimension/seq/include/ops_seq.hpp"

#include <cstddef>
#include <random>
#include <vector>

#include "gutyansky_a_monte_carlo_multi_dimension/common/include/common.hpp"
#include "gutyansky_a_monte_carlo_multi_dimension/common/include/function_registry.hpp"

namespace gutyansky_a_monte_carlo_multi_dimension {

GutyanskyAMonteCarloMultiDimensionSEQ::GutyanskyAMonteCarloMultiDimensionSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  GetInput() = in;
  GetOutput() = 0.0;
}

bool GutyanskyAMonteCarloMultiDimensionSEQ::ValidationImpl() {
  return GetInput().IsValid();
}

bool GutyanskyAMonteCarloMultiDimensionSEQ::PreProcessingImpl() {
  return true;
}

bool GutyanskyAMonteCarloMultiDimensionSEQ::RunImpl() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distr(0.0, 1.0);

  size_t n_points = GetInput().n_points;
  size_t n_dims = GetInput().n_dims;

  double volume = 1.0;

  for (size_t i = 0; i < n_dims; i++) {
    volume *= GetInput().upper_bounds[i] - GetInput().lower_bounds[i];
  }

  FunctionRegistry::IntegralFunction function = GetInput().GetFunction();

  std::vector<double> random_point(n_dims);

  double sum = 0.0;

  for (size_t i = 0; i < n_points; i++) {
    for (size_t j = 0; j < n_dims; j++) {
      double lb = GetInput().lower_bounds[j];
      double rb = GetInput().upper_bounds[j];

      random_point[j] = lb + (distr(gen) * (rb - lb));
    }

    sum += function(random_point);
  }

  GetOutput() = volume * (sum / static_cast<double>(n_points));

  return true;
}

bool GutyanskyAMonteCarloMultiDimensionSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace gutyansky_a_monte_carlo_multi_dimension
