#include "kutuzov_i_simpson_integration/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "kutuzov_i_simpson_integration/common/include/common.hpp"

namespace kutuzov_i_simpson_integration {

KutuzovISimpsonIntegrationMPI::KutuzovISimpsonIntegrationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool KutuzovISimpsonIntegrationMPI::ValidationImpl() {
  int n = std::get<0>(GetInput());
  double x_min = std::get<0>(std::get<1>(GetInput()));
  double x_max = std::get<1>(std::get<1>(GetInput()));
  double y_min = std::get<0>(std::get<2>(GetInput()));
  double y_max = std::get<1>(std::get<2>(GetInput()));
  int function_id = std::get<3>(GetInput());

  if (x_min >= x_max) {
    return false;
  }
  if (y_min >= y_max) {
    return false;
  }
  if (n <= 0 || n % 2 != 0) {
    return false;
  }
  if (function_id <= 0 || function_id > 4) {
    return false;
  }
  return true;
}

bool KutuzovISimpsonIntegrationMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool KutuzovISimpsonIntegrationMPI::RunImpl() {
  int rank = 0;
  int process_count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);

  int n = 0;
  double x_min = 0.0;
  double x_max = 0.0;
  double y_min = 0.0;
  double y_max = 0.0;
  int function_id = 0;

  // To reduce the number of MPI_Bcast calls we can send similar data in bulk
  std::vector<double> data_package(4, 0.0);
  if (rank == 0) {
    n = std::get<0>(GetInput());
    x_min = std::get<1>(GetInput()).first;
    x_max = std::get<1>(GetInput()).second;
    y_min = std::get<2>(GetInput()).first;
    y_max = std::get<2>(GetInput()).second;
    function_id = std::get<3>(GetInput());

    data_package = {x_min, x_max, y_min, y_max};
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&function_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(data_package.data(), 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  x_min = data_package[0];
  x_max = data_package[1];
  y_min = data_package[2];
  y_max = data_package[3];

  int task_per_process = (n + 1) / process_count;
  int tail = (n + 1) - (task_per_process * process_count);
  int task_start = (task_per_process * rank) + std::min(rank, tail);
  int task_end = task_start + task_per_process + (rank < tail ? 1 : 0);

  double step_x = (x_max - x_min) / n;
  double step_y = (y_max - y_min) / n;

  double local_sum = 0.0;
  for (int i = task_start; i < task_end; i++) {
    double x = x_min + (step_x * i);

    for (int j = 0; j <= n; j++) {
      double y = y_min + (step_y * j);
      double a = GetWeight(i, n) * GetWeight(j, n) * CallFunction(function_id, x, y);
      local_sum += a;
    }
  }

  double sum = 0.0;
  MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  sum *= step_x * step_y / 9;
  MPI_Bcast(&sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = sum;

  return true;
}

bool KutuzovISimpsonIntegrationMPI::PostProcessingImpl() {
  return true;
}

double KutuzovISimpsonIntegrationMPI::GetWeight(int i, int n) {
  if (i == 0 || i == n) {
    return 1.0;
  }
  if (i % 2 == 1) {
    return 4.0;
  }
  return 2.0;
}

}  // namespace kutuzov_i_simpson_integration
