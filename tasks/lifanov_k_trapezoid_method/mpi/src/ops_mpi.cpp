#include "lifanov_k_trapezoid_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "lifanov_k_trapezoid_method/common/include/common.hpp"

namespace lifanov_k_trapezoid_method {

namespace {

double Function(double x, double y) {
  return (x * x) + (y * y);
}

double Weight(int ix, int iy, int nx, int ny) {
  double wx = (ix == 0 || ix == nx) ? 0.5 : 1.0;
  double wy = (iy == 0 || iy == ny) ? 0.5 : 1.0;
  return wx * wy;
}

double ComputeLocalSum(int x_start, int x_end, int nx, int ny, double ax, double ay, double hx, double hy) {
  double local_sum = 0.0;

  for (int i = x_start; i <= x_end; ++i) {
    const double x = ax + (i * hx);
    for (int j = 0; j <= ny; ++j) {
      const double y = ay + (j * hy);
      local_sum += Weight(i, j, nx, ny) * Function(x, y);
    }
  }

  return local_sum;
}

}  // namespace

LifanovKTrapezoidMethodMPI::LifanovKTrapezoidMethodMPI(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput() = 0.0;
}

bool LifanovKTrapezoidMethodMPI::ValidationImpl() {
  const auto &in = GetInput();

  if (in.size() != 6) {
    return false;
  }

  const double ax = in[0];
  const double bx = in[1];
  const double ay = in[2];
  const double by = in[3];
  const int nx = static_cast<int>(in[4]);
  const int ny = static_cast<int>(in[5]);

  return bx > ax && by > ay && nx > 0 && ny > 0;
}

bool LifanovKTrapezoidMethodMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool LifanovKTrapezoidMethodMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &in = GetInput();

  const int nx = static_cast<int>(in[4]);
  const int ny = static_cast<int>(in[5]);

  const double hx = (in[1] - in[0]) / nx;
  const double hy = (in[3] - in[2]) / ny;

  const int total_nodes = nx + 1;
  const int base = total_nodes / size;
  const int rem = total_nodes % size;

  const int x_start = (rank * base) + std::min(rank, rem);
  int x_end = x_start + base - 1;
  if (rank < rem) {
    x_end += 1;
  }

  double local_sum = 0.0;
  if (x_start <= x_end) {
    local_sum = ComputeLocalSum(x_start, x_end, static_cast<int>(in[4]), static_cast<int>(in[5]), in[0], in[2], hx, hy);
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = global_sum * hx * hy;
  }

  MPI_Bcast(&GetOutput(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return true;
}

bool LifanovKTrapezoidMethodMPI::PostProcessingImpl() {
  return true;
}

}  // namespace lifanov_k_trapezoid_method
