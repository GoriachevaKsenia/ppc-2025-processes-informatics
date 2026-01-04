#include "liulin_y_integ_mnog_func_monte_carlo/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstdint>
#include <random>

#include "liulin_y_integ_mnog_func_monte_carlo/common/include/common.hpp"

namespace liulin_y_integ_mnog_func_monte_carlo {

LiulinYIntegMnogFuncMonteCarloMPI::LiulinYIntegMnogFuncMonteCarloMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool LiulinYIntegMnogFuncMonteCarloMPI::ValidationImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank != 0) {
    return true;
  }

  const auto &input = GetInput();
  return input.num_points > 0 && input.x_min <= input.x_max && input.y_min <= input.y_max;
}

bool LiulinYIntegMnogFuncMonteCarloMPI::PreProcessingImpl() {
  return true;
}

bool LiulinYIntegMnogFuncMonteCarloMPI::RunImpl() {
  const auto &input = GetInput();
  auto &result = GetOutput();

  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int64_t total_points = input.num_points;
  MPI_Bcast(&total_points, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);

  if (total_points <= 0) {
    result = 0.0;
    return true;
  }

  double area = (input.x_max - input.x_min) * (input.y_max - input.y_min);
  MPI_Bcast(&area, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (area <= 0.0) {
    result = 0.0;
    return true;
  }

  int64_t base_points = total_points / world_size;
  int64_t remainder = total_points % world_size;
  int64_t local_points = base_points + (world_rank < remainder ? 1 : 0);

  std::random_device rd;
  std::mt19937 gen(rd() + world_rank);
  std::uniform_real_distribution<double> dist_x(input.x_min, input.x_max);
  std::uniform_real_distribution<double> dist_y(input.y_min, input.y_max);

  double local_sum = 0.0;
  for (int64_t i = 0; i < local_points; ++i) {
    double x = dist_x(gen);
    double y = dist_y(gen);
    local_sum += input.f(x, y);
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    result = global_sum / static_cast<double>(total_points) * area;
  }

  MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

bool LiulinYIntegMnogFuncMonteCarloMPI::PostProcessingImpl() {
  return true;
}

}  // namespace liulin_y_integ_mnog_func_monte_carlo
