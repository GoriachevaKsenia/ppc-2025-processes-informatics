#include "spichek_d_simpson_integral/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>

// Explicitly include common.hpp to satisfy clang-tidy misc-include-cleaner for InType/OutType
#include "spichek_d_simpson_integral/common/include/common.hpp"

namespace spichek_d_simpson_integral {

namespace {
// Helper to reduce cognitive complexity and avoid nested ternary operators
int GetSimpsonWeight(int index, int n) {
  if (index == 0 || index == n) {
    return 1;
  }
  return (index % 2 == 0) ? 2 : 4;
}
}  // namespace

SpichekDSimpsonIntegralMPI::SpichekDSimpsonIntegralMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SpichekDSimpsonIntegralMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetInput() % 2 == 0);
}

bool SpichekDSimpsonIntegralMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SpichekDSimpsonIntegralMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  if (rank == 0) {
    n = GetInput();
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 0 || n % 2 != 0) {
    GetOutput() = 0;
    return true;
  }

  const double h = 1.0 / n;
  double local_sum = 0.0;

  // Distribute rows (i) among processes
  for (int i = rank; i <= n; i += size) {
    const double x = i * h;
    const int wx = GetSimpsonWeight(i, n);

    for (int j = 0; j <= n; ++j) {
      const double y = j * h;
      const int wy = GetSimpsonWeight(j, n);

      local_sum += wx * wy * (x * x + y * y);
    }
  }

  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  const double result = global_sum * h * h / 9.0;

  GetOutput() = static_cast<OutType>(std::round(result));
  return true;
}

bool SpichekDSimpsonIntegralMPI::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_simpson_integral
