#include "spichek_d_jacobi/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>  // min, max
#include <cmath>      // abs
#include <cstddef>    // size_t
#include <utility>    // std::move
#include <vector>     // std::vector

#include "spichek_d_jacobi/common/include/common.hpp"

namespace spichek_d_jacobi {

SpichekDJacobiMPI::SpichekDJacobiMPI(InType in) : input_(std::move(in)) {
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool SpichekDJacobiMPI::ValidationImpl() {
  const auto &[A, b, eps, max_iter] = input_;
  return !A.empty() && A.size() == b.size() && eps > 0 && max_iter > 0;
}

bool SpichekDJacobiMPI::PreProcessingImpl() {
  GetInput() = input_;
  GetOutput().assign(std::get<1>(input_).size(), 0.0);
  return true;
}

bool SpichekDJacobiMPI::RunImpl() {
  const auto &[A, b, eps, max_iter] = GetInput();
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t n = b.size();
  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);

  size_t chunk = (n + size - 1) / size;
  size_t begin = rank * chunk;
  size_t end = std::min(begin + chunk, n);

  for (int iter = 0; iter < max_iter; ++iter) {
    double local_diff = 0.0;

    for (size_t i = begin; i < end; ++i) {
      double sum = 0.0;
      for (size_t j = 0; j < n; ++j) {
        if (j != i) {
          sum += A[i][j] * x[j];
        }
      }
      x_new[i] = (b[i] - sum) / A[i][i];
      local_diff = std::max(local_diff, std::abs(x_new[i] - x[i]));
    }

    MPI_Allreduce(MPI_IN_PLACE, x_new.data(), static_cast<int>(n), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double global_diff = 0.0;
    MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    x = x_new;
    if (global_diff < eps) {
      break;
    }
  }

  GetOutput() = x;
  return true;
}

bool SpichekDJacobiMPI::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_jacobi
