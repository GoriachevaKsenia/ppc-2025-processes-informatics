#include "dolov_v_monte_carlo_integration/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "dolov_v_monte_carlo_integration/common/include/common.hpp"

namespace dolov_v_monte_carlo_integration {

DolovVMonteCarloIntegrationSEQ::DolovVMonteCarloIntegrationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool DolovVMonteCarloIntegrationSEQ::ValidationImpl() {
  const auto &in = GetInput();
  return in.func && (in.samples_count > 0) && (in.dimension > 0) &&
         (in.center.size() == static_cast<size_t>(in.dimension)) && (in.radius > 0.0);
}

bool DolovVMonteCarloIntegrationSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool DolovVMonteCarloIntegrationSEQ::RunImpl() {
  const auto &in = GetInput();
  const int n_samples = in.samples_count;
  const int dim_count = in.dimension;
  const double radius = in.radius;
  const double r_sq = radius * radius;

  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<double> distribution(-radius, radius);

  double sum_val = 0.0;
  std::vector<double> point(dim_count);

  for (int sample = 0; sample < n_samples; ++sample) {
    for (int dim_idx = 0; dim_idx < dim_count; ++dim_idx) {
      point[dim_idx] = in.center[dim_idx] + distribution(generator);
    }

    bool is_in_domain = true;
    if (in.domain_type == IntegrationDomain::kHyperSphere) {
      double d_sq = 0.0;
      for (int dim_idx = 0; dim_idx < dim_count; ++dim_idx) {
        double diff = point[dim_idx] - in.center[dim_idx];
        d_sq += diff * diff;
      }
      is_in_domain = (d_sq <= r_sq);
    }

    if (is_in_domain) {
      sum_val += in.func(point);
    }
  }

  const double v_cube = std::pow(2.0 * radius, dim_count);
  GetOutput() = v_cube * (sum_val / n_samples);
  return std::isfinite(GetOutput());
}

bool DolovVMonteCarloIntegrationSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace dolov_v_monte_carlo_integration
