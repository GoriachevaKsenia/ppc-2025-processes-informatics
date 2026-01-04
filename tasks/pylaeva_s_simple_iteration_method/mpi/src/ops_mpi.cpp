#include "pylaeva_s_simple_iteration_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cctype>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "pylaeva_s_simple_iteration_method/common/include/common.hpp"

namespace pylaeva_s_simple_iteration_method {
namespace {

constexpr double kEps = 1e-6;
constexpr int kMaxIterations = 10000;

bool DiagonalDominance(const std::vector<double> &a, size_t n) {
  for (size_t i = 0; i < n; i++) {
    double diag = std::fabs(a[(i * n) + i]);  // Модуль диагонального элемента
    double row_sum = 0.0;                     // Сумма модулей недиагональных элементов строки

    for (size_t j = 0; j < n; j++) {
      if (j != i) {
        row_sum += std::fabs(a[(i * n) + j]);
      }
    }
    // Проверка строгого диагонального преобладания:
    // Диагональный элемент должен быть БОЛЬШЕ суммы остальных элементов строки
    if (diag <= row_sum) {
      return false;
    }
  }
  return true;
}

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

void CalculateChunkSizesAndDispls(int proc_num, int n, std::vector<int> &chunk_sizes, std::vector<int> &displs) {
  if (displs.empty()) {
    return;
  }
  int base = n / proc_num;
  int rem = n % proc_num;

  displs[0] = 0;
  for (int i = 0; i < proc_num; ++i) {
    chunk_sizes[i] = base + (i < rem ? 1 : 0);
    if (i > 0) {
      displs[i] = displs[i - 1] + chunk_sizes[i - 1];
    }
  }
}

void CalculateMatrixChunkSizesAndDispls(int proc_num, int n, std::vector<int> &matrix_chunk_sizes,
                                        std::vector<int> &matrix_displs) {
  if (matrix_displs.empty()) {
    return;
  }
  int base = n / proc_num;
  int rem = n % proc_num;

  matrix_displs[0] = 0;
  for (int i = 0; i < proc_num; ++i) {
    int rows = base + (i < rem ? 1 : 0);
    matrix_chunk_sizes[i] = rows * n;
    if (i > 0) {
      matrix_displs[i] = matrix_displs[i - 1] + matrix_chunk_sizes[i - 1];
    }
  }
}

}  // namespace

PylaevaSSimpleIterationMethodMPI::PylaevaSSimpleIterationMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool PylaevaSSimpleIterationMethodMPI::ValidationImpl() {
  const auto &n = std::get<0>(GetInput());
  const auto &a = std::get<1>(GetInput());
  const auto &b = std::get<2>(GetInput());
  return ((n > 0) && (a.size() == n * n) && (b.size() == n) && (DiagonalDominance(a, n)));
}

bool PylaevaSSimpleIterationMethodMPI::PreProcessingImpl() {
  return true;
}

bool PylaevaSSimpleIterationMethodMPI::RunImpl() {
  int proc_num = 0;
  int proc_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  size_t n = 0;
  std::vector<double> a;
  std::vector<double> b;

  if (proc_rank == 0) {
    const auto &input = GetInput();
    n = std::get<0>(input);
    a = std::get<1>(input);
    b = std::get<2>(input);
  }

  MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  std::vector<int> chunk_sizes(proc_num);
  std::vector<int> displs(proc_num);
  CalculateChunkSizesAndDispls(proc_num, static_cast<int>(n), chunk_sizes, displs);

  std::vector<int> matrix_chunk_sizes(proc_num);
  std::vector<int> matrix_displs(proc_num);
  CalculateMatrixChunkSizesAndDispls(proc_num, static_cast<int>(n), matrix_chunk_sizes, matrix_displs);

  int local_rows = chunk_sizes[proc_rank];
  int local_matrix_size = matrix_chunk_sizes[proc_rank];

  std::vector<double> local_a(local_matrix_size);
  std::vector<double> local_b(local_rows);

  MPI_Scatterv(proc_rank == 0 ? a.data() : nullptr, matrix_chunk_sizes.data(), matrix_displs.data(), MPI_DOUBLE,
               local_a.data(), local_matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(proc_rank == 0 ? b.data() : nullptr, chunk_sizes.data(), displs.data(), MPI_DOUBLE, local_b.data(),
               local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int start = displs[proc_rank];
  int count = local_rows;

  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);
  std::vector<double> local_x_new(count, 0.0);

  std::vector<int> recv_counts(proc_num);
  std::vector<int> allgather_displs(proc_num);
  CalculateChunkSizesAndDispls(proc_num, static_cast<int>(n), recv_counts, allgather_displs);

  for (int iter = 0; iter < kMaxIterations; ++iter) {
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

bool PylaevaSSimpleIterationMethodMPI::PostProcessingImpl() {
  return true;
}

}  // namespace pylaeva_s_simple_iteration_method
