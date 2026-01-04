#include "krykov_e_simple_iterations/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cctype>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

#include "krykov_e_simple_iterations/common/include/common.hpp"

namespace krykov_e_simple_iterations {
namespace {

constexpr double kEps = 1e-5;
constexpr int kMaxIter = 10000;

void CalculateLocalXNew(int start, int count, size_t n, const std::vector<double> &local_a,
                        const std::vector<double> &local_b, const std::vector<double> &x,
                        std::vector<double> &local_x_new) {
  for (int i = 0; i < count; ++i) {
    int global_i = start + i;
    double sum = 0.0;
    for (size_t j = 0; j < n; ++j) {
      if (std::cmp_not_equal(j, global_i)) {
        sum += local_a[(i * n) + j] * x[j];
      }
    }
    local_x_new[i] = (local_b[i] - sum) / local_a[(i * n) + global_i];
  }
}

double CalculateLocalNorm(int start, int count, const std::vector<double> &x_new, const std::vector<double> &x) {
  double local_norm = 0.0;
  for (int i = 0; i < count; ++i) {
    int gi = start + i;
    double diff = x_new[gi] - x[gi];
    local_norm += diff * diff;
  }
  return local_norm;
}

void CalculateChunkSizesAndDispls(int size, int n, std::vector<int> &chunk_sizes, std::vector<int> &displs) {
  if (displs.empty()) {
    return;
  }
  int base = n / size;
  int rem = n % size;

  displs[0] = 0;
  for (int i = 0; i < size; ++i) {
    chunk_sizes[i] = base + (i < rem ? 1 : 0);
    if (i > 0) {
      displs[i] = displs[i - 1] + chunk_sizes[i - 1];
    }
  }
}

void CalculateMatrixChunkSizesAndDispls(int size, int n, std::vector<int> &matrix_chunk_sizes,
                                        std::vector<int> &matrix_displs) {
  if (matrix_displs.empty()) {
    return;
  }
  int base = n / size;
  int rem = n % size;

  matrix_displs[0] = 0;
  for (int i = 0; i < size; ++i) {
    int rows = base + (i < rem ? 1 : 0);
    matrix_chunk_sizes[i] = rows * n;
    if (i > 0) {
      matrix_displs[i] = matrix_displs[i - 1] + matrix_chunk_sizes[i - 1];
    }
  }
}

}  // namespace

KrykovESimpleIterationsMPI::KrykovESimpleIterationsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrykovESimpleIterationsMPI::ValidationImpl() {
  const auto &[n, a, b] = GetInput();
  return n > 0 && a.size() == n * n && b.size() == n;
}

bool KrykovESimpleIterationsMPI::PreProcessingImpl() {
  return true;
}

bool KrykovESimpleIterationsMPI::RunImpl() {
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  size_t n = 0;
  std::vector<double> a;
  std::vector<double> b;

  if (rank == 0) {
    const auto &input = GetInput();
    n = std::get<0>(input);
    a = std::get<1>(input);
    b = std::get<2>(input);
  }

  MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  std::vector<int> chunk_sizes(size);
  std::vector<int> displs(size);
  CalculateChunkSizesAndDispls(size, static_cast<int>(n), chunk_sizes, displs);

  std::vector<int> matrix_chunk_sizes(size);
  std::vector<int> matrix_displs(size);
  CalculateMatrixChunkSizesAndDispls(size, static_cast<int>(n), matrix_chunk_sizes, matrix_displs);

  int local_rows = chunk_sizes[rank];
  int local_matrix_size = matrix_chunk_sizes[rank];

  std::vector<double> local_a(local_matrix_size);
  std::vector<double> local_b(local_rows);

  MPI_Scatterv(rank == 0 ? a.data() : nullptr, matrix_chunk_sizes.data(), matrix_displs.data(), MPI_DOUBLE,
               local_a.data(), local_matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? b.data() : nullptr, chunk_sizes.data(), displs.data(), MPI_DOUBLE, local_b.data(),
               local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int start = displs[rank];
  int count = local_rows;

  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);
  std::vector<double> local_x_new(count, 0.0);

  std::vector<int> recv_counts(size);
  std::vector<int> allgather_displs(size);
  CalculateChunkSizesAndDispls(size, static_cast<int>(n), recv_counts, allgather_displs);

  for (int iter = 0; iter < kMaxIter; ++iter) {
    CalculateLocalXNew(start, count, n, local_a, local_b, x, local_x_new);

    MPI_Allgatherv(local_x_new.data(), count, MPI_DOUBLE, x_new.data(), recv_counts.data(), allgather_displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    double local_norm = CalculateLocalNorm(start, count, x_new, x);
    double global_norm = 0.0;
    MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    x = x_new;

    if (std::sqrt(global_norm) < kEps) {
      break;
    }
  }

  GetOutput() = x;

  return true;
}

bool KrykovESimpleIterationsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace krykov_e_simple_iterations
