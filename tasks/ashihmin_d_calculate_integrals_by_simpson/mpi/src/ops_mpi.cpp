#include "ashihmin_d_calculate_integrals_by_simpson/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
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

double ComputeLocalSum(int start_index, int end_index, int dimensions, int partitions,
                       const std::vector<double> &step_sizes, const std::vector<double> &left_bounds) {
  double local_sum = 0.0;

  std::vector<int> indices(static_cast<std::size_t>(dimensions), 0);

  for (int first_index = start_index; first_index <= end_index; ++first_index) {
    indices[0] = first_index;

    int inner_points_count = 1;
    for (int dim = 1; dim < dimensions; ++dim) {
      inner_points_count *= (partitions + 1);
    }

    for (int inner_index = 0; inner_index < inner_points_count; ++inner_index) {
      int temp_index = inner_index;
      for (int dim = 1; dim < dimensions; ++dim) {
        indices[dim] = temp_index % (partitions + 1);
        temp_index /= (partitions + 1);
      }

      const double weight = CalculateWeightCoefficient(indices, partitions);
      const auto point = CalculatePoint(indices, step_sizes, left_bounds);
      local_sum += weight * Function(point);
    }
  }

  return local_sum;
}

}  // namespace

AshihminDCalculateIntegralsBySimpsonMPI::AshihminDCalculateIntegralsBySimpsonMPI(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput() = 0.0;
}

bool AshihminDCalculateIntegralsBySimpsonMPI::ValidationImpl() {
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

bool AshihminDCalculateIntegralsBySimpsonMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

double AshihminDCalculateIntegralsBySimpsonMPI::IntegrandFunction(const std::vector<double> &coordinates) {
  return Function(coordinates);
}

bool AshihminDCalculateIntegralsBySimpsonMPI::RunImpl() {
  int process_rank = 0;
  int process_count = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);

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

  const int total_nodes = partitions + 1;
  const int chunk_size = total_nodes / process_count;
  const int remainder = total_nodes % process_count;

  int start_index = (process_rank * chunk_size) + std::min(process_rank, remainder);
  int end_index = start_index + chunk_size - 1;
  if (process_rank < remainder) {
    end_index += 1;
  }

  double local_sum = 0.0;
  if (start_index <= end_index) {
    local_sum = ComputeLocalSum(start_index, end_index, dimensions, partitions, step_sizes, input.left_bounds);
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  double volume_element = 1.0;
  for (double step : step_sizes) {
    volume_element *= step;
  }

  if (process_rank == 0) {
    GetOutput() = global_sum * volume_element / std::pow(3.0, dimensions);
  }

  MPI_Bcast(&GetOutput(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return true;
}

bool AshihminDCalculateIntegralsBySimpsonMPI::PostProcessingImpl() {
  return true;
}

}  // namespace ashihmin_d_calculate_integrals_by_simpson
