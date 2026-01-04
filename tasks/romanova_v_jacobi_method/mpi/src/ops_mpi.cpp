#include "romanova_v_jacobi_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "romanova_v_jacobi_method/common/include/common.hpp"

namespace romanova_v_jacobi_method {

RomanovaVJacobiMethodMPI::RomanovaVJacobiMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput();
}

bool RomanovaVJacobiMethodMPI::ValidationImpl() {
  bool status = true;

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    std::vector<OutType> a = std::get<1>(GetInput());

    OutType x = std::get<0>(GetInput());
    OutType b = std::get<2>(GetInput());
    status = status && !a.empty();
    for (size_t i = 0; i < a.size(); i++) {
      status = status && (a.size() == a[i].size());
    }

    status = status && IsDiagonallyDominant(a);

    status = status && (a.size() == x.size());
    status = status && (a.size() == b.size());
  }

  MPI_Bcast(&status, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return status;
}

bool RomanovaVJacobiMethodMPI::PreProcessingImpl() {
  int rank = 0;
  int n = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n);

  std::vector<int> send_counts_a(n);
  vector_counts_ = std::vector<int>(n);

  std::vector<int> displs_scatt_a(n);
  vector_displs_ = std::vector<int>(n);

  if (rank == 0) {
    n_ = std::get<1>(GetInput()).size();
  }

  MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  x_ = OutType(n_);

  if (rank == 0) {
    std::vector<OutType> temp_a;
    std::tie(x_, temp_a, b_, eps_, maxIterations_) = GetInput();

    for (const auto &vec : temp_a) {
      A_.insert(A_.end(), vec.begin(), vec.end());
    }
  }

  MPI_Bcast(&eps_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&maxIterations_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  size_t delta = n_ / n;
  size_t extra = n_ % n;

  local_n_ = delta + (std::cmp_less(rank, extra) ? 1 : 0);
  st_row_ = (rank * delta) + (std::cmp_less(rank, extra) ? rank : extra);

  if (rank == 0) {
    send_counts_a = std::vector<int>(n, static_cast<int>(delta * n_));
    vector_counts_ = std::vector<int>(n, static_cast<int>(delta));
    for (size_t i = 0; i < extra; i++) {
      send_counts_a[i] += static_cast<int>(n_);
      vector_counts_[i]++;
    }

    for (int i = 1; i < n; i++) {
      displs_scatt_a[i] = displs_scatt_a[i - 1] + send_counts_a[i - 1];
      vector_displs_[i] = vector_displs_[i - 1] + vector_counts_[i - 1];
    }
  }

  MPI_Bcast(vector_counts_.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(vector_displs_.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

  OutType local_data = OutType(local_n_ * n_);
  OutType local_b = OutType(local_n_);

  MPI_Bcast(x_.data(), static_cast<int>(n_), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? A_.data() : nullptr, send_counts_a.data(), displs_scatt_a.data(), MPI_DOUBLE,
               local_data.data(), static_cast<int>(local_data.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? b_.data() : nullptr, vector_counts_.data(), vector_displs_.data(), MPI_DOUBLE,
               local_b.data(), static_cast<int>(local_b.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  A_ = std::move(local_data);
  b_ = std::move(local_b);

  return true;
}

bool RomanovaVJacobiMethodMPI::RunImpl() {
  size_t k = 0;
  OutType prev(x_.size(), 0.0);
  OutType local_x(local_n_);
  double diff = 0.0;
  double glob_diff = eps_;
  while (glob_diff >= eps_ && k < maxIterations_) {
    diff = 0.0;
    prev = x_;
    for (size_t i = 0; i < local_n_; i++) {
      double sum = 0.0;
      for (size_t j = 0; j < n_; j++) {
        sum += ((st_row_ + i) != j ? A_[(i * n_) + j] * prev[j] : 0);
      }
      local_x[i] = (b_[i] - sum) / A_[(i * (n_ + 1)) + st_row_];

      diff = std::max(diff, abs(local_x[i] - prev[st_row_ + i]));
    }

    MPI_Allgatherv(local_x.data(), static_cast<int>(local_n_), MPI_DOUBLE, x_.data(), vector_counts_.data(),
                   vector_displs_.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    MPI_Allreduce(&diff, &glob_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    k++;
  }
  return true;
}

bool RomanovaVJacobiMethodMPI::PostProcessingImpl() {
  GetOutput() = x_;
  return true;
}

bool RomanovaVJacobiMethodMPI::IsDiagonallyDominant(const std::vector<OutType> &matrix) {
  for (size_t i = 0; i < matrix.size(); i++) {
    double sum = 0.0;
    for (size_t j = 0; j < matrix[i].size(); j++) {
      if (i != j) {
        sum += matrix[i][j];
      }
    }
    if (matrix[i][i] <= sum) {
      return false;
    }
  }
  return true;
}

}  // namespace romanova_v_jacobi_method
