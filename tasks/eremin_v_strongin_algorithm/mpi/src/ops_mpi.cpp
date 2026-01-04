#include "eremin_v_strongin_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <tuple>
#include <vector>

#include "eremin_v_strongin_algorithm/common/include/common.hpp"

namespace eremin_v_strongin_algorithm {

EreminVStronginAlgorithmMPI::EreminVStronginAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool EreminVStronginAlgorithmMPI::ValidationImpl() {
  auto &input = GetInput();

  double lower_bound = std::get<0>(input);
  double upper_bound = std::get<1>(input);
  double epsilon = std::get<2>(input);
  int max_iters = std::get<3>(input);

  return (lower_bound < upper_bound) && (epsilon > 0.0 && epsilon <= (upper_bound - lower_bound)) &&
         (max_iters > 0 && max_iters <= 100000000) && (lower_bound >= -1e9 && lower_bound <= 1e9) &&
         (upper_bound >= -1e9 && upper_bound <= 1e9) && (GetOutput() == 0);
}

bool EreminVStronginAlgorithmMPI::PreProcessingImpl() {
  return true;
}

double EreminVStronginAlgorithmMPI::CalculateLipschitzEstimate(int rank, int size,
                                                               const std::vector<double> &search_points,
                                                               const std::vector<double> &function_values) {
  double lipschitz_estimate = 0.0;
  for (std::size_t i = 1 + static_cast<std::size_t>(rank); i < search_points.size();
       i += static_cast<std::size_t>(size)) {
    double interval_width = search_points[i] - search_points[i - 1];
    double value_difference = std::abs(function_values[i] - function_values[i - 1]);
    double current_slope = value_difference / interval_width;
    lipschitz_estimate = std::max(current_slope, lipschitz_estimate);
  }

  double global_lipschitz_estimate = 0.0;
  MPI_Allreduce(&lipschitz_estimate, &global_lipschitz_estimate, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  return global_lipschitz_estimate;
}

EreminVStronginAlgorithmMPI::IntervalCharacteristic EreminVStronginAlgorithmMPI::FindBestInterval(
    int rank, int size, const std::vector<double> &search_points, const std::vector<double> &function_values,
    double m_parameter) {
  IntervalCharacteristic local{.value = -1e18, .index = 1};

  for (std::size_t i = 1 + static_cast<std::size_t>(rank); i < search_points.size();
       i += static_cast<std::size_t>(size)) {
    double interval_width = search_points[i] - search_points[i - 1];
    double value_difference = function_values[i] - function_values[i - 1];

    double characteristic = (m_parameter * interval_width) +
                            ((value_difference * value_difference) / (m_parameter * interval_width)) -
                            (2.0 * (function_values[i] + function_values[i - 1]));

    if (characteristic > local.value) {
      local.value = characteristic;
      local.index = static_cast<int>(i);
    }
  }

  IntervalCharacteristic global{};
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
  return global;
}

bool EreminVStronginAlgorithmMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double lower_bound = 0.0;
  double upper_bound = 0.0;
  double epsilon = 0.0;
  int max_iterations = 0;

  if (rank == 0) {
    auto &input = GetInput();
    lower_bound = std::get<0>(input);
    upper_bound = std::get<1>(input);
    epsilon = std::get<2>(input);
    max_iterations = std::get<3>(input);
  }

  MPI_Bcast(&lower_bound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&upper_bound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&max_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

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

    double lipschitz_estimate = CalculateLipschitzEstimate(rank, size, search_points, function_values);

    double m_parameter = (lipschitz_estimate > 0.0) ? r_coefficient * lipschitz_estimate : 1.0;

    auto best_interval = FindBestInterval(rank, size, search_points, function_values, m_parameter);
    int best_interval_index = best_interval.index;

    double left_point = search_points[best_interval_index - 1];
    double right_point = search_points[best_interval_index];
    double left_value = function_values[best_interval_index - 1];
    double right_value = function_values[best_interval_index];

    double new_point = 0.0;
    double new_value = 0.0;

    if (rank == 0) {
      new_point = (0.5 * (left_point + right_point)) - ((right_value - left_value) / (2.0 * m_parameter));

      if (new_point <= left_point || new_point >= right_point) {
        new_point = 0.5 * (left_point + right_point);
      }

      new_value = objective_function(new_point);
    }
    MPI_Bcast(&new_point, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&new_value, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    search_points.insert(search_points.begin() + static_cast<std::ptrdiff_t>(best_interval_index), new_point);
    function_values.insert(function_values.begin() + static_cast<std::ptrdiff_t>(best_interval_index), new_value);

    max_interval_width = 0.0;
    for (std::size_t i = 1 + static_cast<std::size_t>(rank); i < search_points.size();
         i += static_cast<std::size_t>(size)) {
      double current_width = search_points[i] - search_points[i - 1];
      max_interval_width = std::max(current_width, max_interval_width);
    }
    double local_max_interval_width = max_interval_width;
    MPI_Allreduce(&local_max_interval_width, &max_interval_width, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  }
  GetOutput() = *std::ranges::min_element(function_values);
  return true;
}

bool EreminVStronginAlgorithmMPI::PostProcessingImpl() {
  return true;
}

}  // namespace eremin_v_strongin_algorithm
