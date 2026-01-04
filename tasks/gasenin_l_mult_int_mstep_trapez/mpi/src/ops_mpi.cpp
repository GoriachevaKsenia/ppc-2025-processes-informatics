#include "gasenin_l_mult_int_mstep_trapez/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>

#include "gasenin_l_mult_int_mstep_trapez/common/include/common.hpp"

namespace gasenin_l_mult_int_mstep_trapez {

GaseninLMultIntMstepTrapezMPI::GaseninLMultIntMstepTrapezMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool GaseninLMultIntMstepTrapezMPI::ValidationImpl() {
  return GetInput().n_steps > 0 && GetInput().x2 > GetInput().x1 && GetInput().y2 > GetInput().y1;
}

bool GaseninLMultIntMstepTrapezMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

namespace {
template <typename Func>
double RunKernel(const TaskData &data, int rank, int size, const Func &f) {
  double hx = (data.x2 - data.x1) / data.n_steps;
  double hy = (data.y2 - data.y1) / data.n_steps;

  int total_nodes_x = data.n_steps + 1;

  int count = total_nodes_x / size;
  int remainder = total_nodes_x % size;

  int start_i = (rank * count) + std::min(rank, remainder);
  int end_i = start_i + count + (rank < remainder ? 1 : 0);

  double local_sum = 0.0;

  for (int i = start_i; i < end_i; ++i) {
    double x = data.x1 + (i * hx);
    double weight_x = (i == 0 || i == data.n_steps) ? 0.5 : 1.0;

    for (int j = 0; j <= data.n_steps; ++j) {
      double y = data.y1 + (j * hy);
      double weight_y = (j == 0 || j == data.n_steps) ? 0.5 : 1.0;

      local_sum += f(x, y) * weight_x * weight_y;
    }
  }

  return local_sum * hx * hy;
}
}  // namespace

bool GaseninLMultIntMstepTrapezMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  TaskData data{};
  if (rank == 0) {
    data = GetInput();
  }

  MPI_Bcast(&data.n_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&data.func_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&data.x1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&data.x2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&data.y1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&data.y2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double local_result = 0.0;

  switch (data.func_id) {
    case 0:
      local_result = RunKernel(data, rank, size, [](double x, double y) { return x + y; });
      break;
    case 1:
      local_result = RunKernel(data, rank, size, [](double x, double y) { return (x * x) + (y * y); });
      break;
    case 2:
      local_result = RunKernel(data, rank, size, [](double x, double y) { return std::sin(x) * std::cos(y); });
      break;
    case 3:
      local_result = RunKernel(data, rank, size, [](double x, double y) { return std::exp(x + y); });
      break;
    case 4:
      local_result = RunKernel(data, rank, size, [](double x, double y) { return std::sqrt((x * x) + (y * y)); });
      break;
    default:
      local_result = RunKernel(data, rank, size, [](double x, double y) {
        (void)x;
        (void)y;
        return 1.0;
      });
      break;
  }

  double global_result = 0.0;
  MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(&global_result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = global_result;

  return true;
}

bool GaseninLMultIntMstepTrapezMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gasenin_l_mult_int_mstep_trapez
