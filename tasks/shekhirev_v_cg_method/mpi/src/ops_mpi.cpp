#include "../include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "../../common/include/common.hpp"

namespace shekhirev_v_cg_method_mpi {

ConjugateGradientMPI::ConjugateGradientMPI(const shekhirev_v_cg_method::InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ConjugateGradientMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const int n = GetInput().n;
    return n > 0 && GetInput().A.size() == static_cast<size_t>(n) * n && GetInput().b.size() == static_cast<size_t>(n);
  }
  return true;
}

bool ConjugateGradientMPI::PreProcessingImpl() {
  return true;
}

std::vector<double> ConjugateGradientMPI::LocalMultiply(const std::vector<double> &local_a,
                                                        const std::vector<double> &global_p, int local_n, int n) {
  std::vector<double> res(local_n, 0.0);
  for (int i = 0; i < local_n; ++i) {
    for (int j = 0; j < n; ++j) {
      size_t index = (static_cast<size_t>(i) * n) + j;
      res[i] += local_a[index] * global_p[j];
    }
  }
  return res;
}

double ConjugateGradientMPI::LocalDotProduct(const std::vector<double> &a, const std::vector<double> &b) {
  double res = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    res += (a[i] * b[i]);
  }
  return res;
}

bool ConjugateGradientMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  if (rank == 0) {
    n = GetInput().n;
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int count_per_proc = n / size;
  const int remainder = n % size;

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  std::vector<int> matrix_counts(size);
  std::vector<int> matrix_displs(size);

  int offset = 0;
  for (int i = 0; i < size; ++i) {
    counts[i] = count_per_proc + (i < remainder ? 1 : 0);
    displs[i] = offset;
    matrix_counts[i] = counts[i] * n;
    matrix_displs[i] = offset * n;
    offset += counts[i];
  }

  const int local_n = counts[rank];
  std::vector<double> local_a(static_cast<size_t>(local_n) * n);
  std::vector<double> local_b(local_n);

  const auto &input_a = GetInput().A;
  const auto &input_bb = GetInput().b;

  MPI_Scatterv(input_a.data(), matrix_counts.data(), matrix_displs.data(), MPI_DOUBLE, local_a.data(), local_n * n,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(input_bb.data(), counts.data(), displs.data(), MPI_DOUBLE, local_b.data(), local_n, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  std::vector<double> local_x(local_n, 0.0);
  std::vector<double> local_r = local_b;
  std::vector<double> local_p = local_r;
  std::vector<double> global_p(n);

  double local_rs_old = LocalDotProduct(local_r, local_r);
  double global_rs_old = 0.0;
  MPI_Allreduce(&local_rs_old, &global_rs_old, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  const int max_iter = n * 2;
  const double epsilon = 1e-10;

  for (int k = 0; k < max_iter; ++k) {
    MPI_Allgatherv(local_p.data(), local_n, MPI_DOUBLE, global_p.data(), counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    std::vector<double> local_ap = LocalMultiply(local_a, global_p, local_n, n);
    double local_p_ap = LocalDotProduct(local_p, local_ap);

    double global_p_ap = 0.0;
    MPI_Allreduce(&local_p_ap, &global_p_ap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    const double alpha = global_rs_old / global_p_ap;

    double local_rs_new = 0.0;
    for (int i = 0; i < local_n; ++i) {
      local_x[i] += (alpha * local_p[i]);
      local_r[i] -= (alpha * local_ap[i]);
      local_rs_new += (local_r[i] * local_r[i]);
    }

    double global_rs_new = 0.0;
    MPI_Allreduce(&local_rs_new, &global_rs_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (std::sqrt(global_rs_new) < epsilon) {
      break;
    }

    const double beta = global_rs_new / global_rs_old;
    global_rs_old = global_rs_new;

    for (int i = 0; i < local_n; ++i) {
      local_p[i] = local_r[i] + (beta * local_p[i]);
    }
  }

  if (rank == 0) {
    GetOutput().resize(n);
  }

  MPI_Gatherv(local_x.data(), local_n, MPI_DOUBLE, GetOutput().data(), counts.data(), displs.data(), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

  return true;
}

bool ConjugateGradientMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shekhirev_v_cg_method_mpi
