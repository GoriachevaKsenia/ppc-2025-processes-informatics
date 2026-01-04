#include "ashihmin_d_calculate_integrals_by_simpson/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "ashihmin_d_calculate_integrals_by_simpson/common/include/common.hpp"

namespace ashihmin_d_calculate_integrals_by_simpson {

namespace {

double Function(const std::vector<double> &coordinates) {
  double total = 0.0;
  for (double value : coordinates) {
    total += value * value;
  }
  return total;
}

double CalculateWeightCoefficient(const std::vector<int> &indices, int partitions) {
  double weight_coefficient = 1.0;
  for (int node_index : indices) {
    if (node_index == 0 || node_index == partitions) {
      weight_coefficient *= 1.0;
    } else if (node_index % 2 == 0) {
      weight_coefficient *= 2.0;
    } else {
      weight_coefficient *= 4.0;
    }
  }
  return weight_coefficient;
}

std::vector<double> CalculatePoint(const std::vector<int> &indices, const std::vector<double> &step_sizes,
                                   const std::vector<double> &left_bounds) {
  std::vector<double> point(indices.size());
  for (std::size_t dim = 0; dim < indices.size(); ++dim) {
    point[dim] = left_bounds[dim] + (indices[dim] * step_sizes[dim]);
  }
  return point;
}

}  // namespace

AshihminDCalculateIntegralsBySimpsonSEQ::AshihminDCalculateIntegralsBySimpsonSEQ(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput() = 0.0;
}

bool AshihminDCalculateIntegralsBySimpsonSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.left_bounds.size() != input.right_bounds.size()) {
    return false;
  }
  if (input.partitions <= 0 || input.partitions % 2 != 0) {
    return false;
  }
  for (std::size_t index = 0; index < input.left_bounds.size(); ++index) {
    if (input.right_bounds[index] <= input.left_bounds[index]) {
      return false;
    }
  }
  return true;
}

bool AshihminDCalculateIntegralsBySimpsonSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

double AshihminDCalculateIntegralsBySimpsonSEQ::IntegrandFunction(const std::vector<double> &coordinates) {
  return Function(coordinates);
}

bool AshihminDCalculateIntegralsBySimpsonSEQ::RunImpl() {
  const auto &input = GetInput();
  const int dimensions = static_cast<int>(input.left_bounds.size());
  const int partitions = input.partitions;

  if (dimensions == 0) {
    GetOutput() = 0.0;
    return true;
  }

  std::vector<double> step_sizes(static_cast<std::size_t>(dimensions));
  for (int dim = 0; dim < dimensions; ++dim) {
    step_sizes[static_cast<std::size_t>(dim)] = (input.right_bounds[dim] - input.left_bounds[dim]) / partitions;
  }

  std::vector<int> indices(static_cast<std::size_t>(dimensions), 0);
  int total_points = 1;
  for (int dim = 0; dim < dimensions; ++dim) {
    total_points *= (partitions + 1);
  }

  double total_sum = 0.0;
  for (int point_idx = 0; point_idx < total_points; ++point_idx) {
    int temp = point_idx;
    for (int dim = 0; dim < dimensions; ++dim) {
      indices[static_cast<std::size_t>(dim)] = temp % (partitions + 1);
      temp /= (partitions + 1);
    }

    const double weight = CalculateWeightCoefficient(indices, partitions);
    const auto point = CalculatePoint(indices, step_sizes, input.left_bounds);
    total_sum += weight * Function(point);
  }

  double volume_element = 1.0;
  for (double step : step_sizes) {
    volume_element *= step;
  }

  GetOutput() = total_sum * volume_element / std::pow(3.0, dimensions);
  return true;
}

bool AshihminDCalculateIntegralsBySimpsonSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace ashihmin_d_calculate_integrals_by_simpson
