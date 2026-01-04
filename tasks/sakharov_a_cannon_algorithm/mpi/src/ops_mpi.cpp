#include "sakharov_a_cannon_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "sakharov_a_cannon_algorithm/common/include/common.hpp"

namespace sakharov_a_cannon_algorithm {

namespace {

void LocalMultiply(const std::vector<double> &a_block, const std::vector<double> &b_block, std::vector<double> &c_block,
                   int local_rows, int k_dim, int local_cols) {
  for (int ii = 0; ii < local_rows; ++ii) {
    for (int kk = 0; kk < k_dim; ++kk) {
      double a_val = a_block[Idx(k_dim, ii, kk)];
      for (int jj = 0; jj < local_cols; ++jj) {
        c_block[Idx(local_cols, ii, jj)] += a_val * b_block[Idx(local_cols, kk, jj)];
      }
    }
  }
}

}  // namespace

SakharovACannonAlgorithmMPI::SakharovACannonAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SakharovACannonAlgorithmMPI::ValidationImpl() {
  return IsValidInput(GetInput());
}

bool SakharovACannonAlgorithmMPI::PreProcessingImpl() {
  const auto &input = GetInput();
  auto out_size = static_cast<std::size_t>(input.rows_a) * static_cast<std::size_t>(input.cols_b);
  GetOutput().assign(out_size, 0.0);
  return true;
}

bool SakharovACannonAlgorithmMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &input = GetInput();
  const int m = input.rows_a;
  const int k = input.cols_a;
  const int n = input.cols_b;

  int base_rows = m / world_size;
  int extra_rows = m % world_size;

  std::vector<int> row_counts(world_size);
  std::vector<int> row_displs(world_size);
  int offset = 0;
  for (int idx = 0; idx < world_size; ++idx) {
    row_counts[idx] = base_rows + (idx < extra_rows ? 1 : 0);
    row_displs[idx] = offset;
    offset += row_counts[idx];
  }

  int local_rows = row_counts[rank];

  std::vector<double> local_a(static_cast<std::size_t>(local_rows) * static_cast<std::size_t>(k));
  std::vector<double> local_c(static_cast<std::size_t>(local_rows) * static_cast<std::size_t>(n), 0.0);

  std::vector<int> send_counts_a(world_size);
  std::vector<int> displs_a(world_size);
  for (int idx = 0; idx < world_size; ++idx) {
    send_counts_a[idx] = row_counts[idx] * k;
    displs_a[idx] = row_displs[idx] * k;
  }

  MPI_Scatterv(input.a.data(), send_counts_a.data(), displs_a.data(), MPI_DOUBLE, local_a.data(), local_rows * k,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> b_data(static_cast<std::size_t>(k) * static_cast<std::size_t>(n));
  if (rank == 0) {
    b_data = input.b;
  }
  MPI_Bcast(b_data.data(), k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  LocalMultiply(local_a, b_data, local_c, local_rows, k, n);

  std::vector<int> recv_counts_c(world_size);
  std::vector<int> displs_c(world_size);
  for (int idx = 0; idx < world_size; ++idx) {
    recv_counts_c[idx] = row_counts[idx] * n;
    displs_c[idx] = row_displs[idx] * n;
  }

  MPI_Gatherv(local_c.data(), local_rows * n, MPI_DOUBLE, GetOutput().data(), recv_counts_c.data(), displs_c.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Bcast(GetOutput().data(), m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

bool SakharovACannonAlgorithmMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sakharov_a_cannon_algorithm
