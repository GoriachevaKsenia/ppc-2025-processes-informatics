#include "kiselev_i_gauss_method_horizontal_tape_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "kiselev_i_gauss_method_horizontal_tape_scheme/common/include/common.hpp"

namespace kiselev_i_gauss_method_horizontal_tape_scheme {

KiselevITestTaskMPI::KiselevITestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &buf = GetInput();
  InType tmp(in);
  buf.swap(tmp);
  GetOutput().clear();
}

bool KiselevITestTaskMPI::ValidationImpl() {
  const auto &a_vector = std::get<0>(GetInput());
  const auto &b_vector = std::get<1>(GetInput());
  return !a_vector.empty() && a_vector.size() == b_vector.size() && GetOutput().empty();
}

bool KiselevITestTaskMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

namespace {

void MakePartition(int proc, int num, std::vector<int> &cnt, std::vector<int> &disp) {
  cnt.assign(proc, 0);
  disp.assign(proc, 0);

  const int q_coef = num / proc;
  const int r_coef = num % proc;
  int pos = 0;

  for (int index = 0; index < proc; ++index) {
    cnt[index] = q_coef + (index < r_coef ? 1 : 0);
    disp[index] = pos;
    pos += cnt[index];
  }
}

int OwnerOf(int row, const std::vector<int> &cnt, const std::vector<int> &disp) {
  for (std::size_t proc = 0; proc < cnt.size(); ++proc) {
    if (row >= disp[proc] && row < disp[proc] + cnt[proc]) {
      return static_cast<int>(proc);
    }
  }
  return 0;
}

bool BuildPivotRow(int k_index, int band, int w_coef, int rank, int owner, int row0,
                   const std::vector<double> &a_vector, const std::vector<double> &b_vector, std::vector<double> &pivot,
                   double &rhs_pivot) {
  if (rank != owner) {
    return true;
  }

  const int local_k = k_index - row0;
  const auto row_offset = static_cast<std::size_t>(local_k) * static_cast<std::size_t>(w_coef);
  const double *row = &a_vector[row_offset];

  const double diag = row[band];
  if (diag == 0.0) {
    return false;
  }

  const int right = static_cast<int>(pivot.size()) - 1 + k_index;
  for (int j = k_index; j <= right; ++j) {
    pivot[j - k_index] = row[j - (k_index - band)];
  }

  rhs_pivot = b_vector[static_cast<std::size_t>(local_k)];
  return true;
}

bool ApplyElimination(int k_index, int band, int w_coef, int row0, int local_rows, const std::vector<double> &pivot,
                      double rhs_pivot, std::vector<double> &a_vector, std::vector<double> &b_vector) {
  const double diag = pivot[0];
  if (diag == 0.0) {
    return false;
  }

  const int right = k_index + static_cast<int>(pivot.size()) - 1;
  const int row_begin = std::max(row0, k_index + 1);
  const int row_end = std::min(row0 + local_rows - 1, k_index + band);

  for (int index = row_begin; index <= row_end; ++index) {
    const int local_i = index - row0;
    const auto row_offset = static_cast<std::size_t>(local_i) * static_cast<std::size_t>(w_coef);
    double *row = &a_vector[row_offset];

    const int col = k_index - (index - band);
    if (col < 0 || col >= w_coef) {
      continue;
    }

    const double multiplier = row[col] / diag;
    row[col] = 0.0;

    for (int j = k_index + 1; j <= right; ++j) {
      const int idx = j - (index - band);
      if (idx >= 0 && idx < w_coef) {
        row[idx] -= multiplier * pivot[j - k_index];
      }
    }

    b_vector[static_cast<std::size_t>(local_i)] -= multiplier * rhs_pivot;
  }

  return true;
}

bool EliminateForward(int num, int band, int w_coef, int rank, int row0, int local_rows, const std::vector<int> &cnt,
                      const std::vector<int> &disp, std::vector<double> &a_vector, std::vector<double> &b_vector) {
  for (int k_index = 0; k_index < num; ++k_index) {
    const int owner = OwnerOf(k_index, cnt, disp);
    const int right = std::min(num - 1, k_index + band);
    const int length = right - k_index + 1;

    std::vector<double> pivot(static_cast<std::size_t>(length), 0.0);
    double rhs_pivot = 0.0;

    if (!BuildPivotRow(k_index, band, w_coef, rank, owner, row0, a_vector, b_vector, pivot, rhs_pivot)) {
      return false;
    }

    MPI_Bcast(pivot.data(), length, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    MPI_Bcast(&rhs_pivot, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

    if (!ApplyElimination(k_index, band, w_coef, row0, local_rows, pivot, rhs_pivot, a_vector, b_vector)) {
      return false;
    }
  }

  return true;
}

bool EliminateBackward(int num, int band, int w_coef, int rank, int row0, const std::vector<int> &cnt,
                       const std::vector<int> &disp, const std::vector<double> &a_vector,
                       const std::vector<double> &b_vector, std::vector<double> &x_vector) {
  x_vector.assign(num, 0.0);

  for (int k_index = num - 1; k_index >= 0; --k_index) {
    const int owner = OwnerOf(k_index, cnt, disp);
    const int right = std::min(num - 1, k_index + band);

    double val = 0.0;

    if (rank == owner) {
      const int local_k = k_index - row0;
      const double *row = &a_vector[static_cast<std::size_t>(local_k) * static_cast<std::size_t>(w_coef)];

      double sum = 0.0;
      for (int j_index = k_index + 1; j_index <= right; ++j_index) {
        const int idx = j_index - (k_index - band);
        if (idx >= 0 && idx < w_coef) {
          sum += row[idx] * x_vector[j_index];
        }
      }

      const double diag = row[band];
      if (diag == 0.0) {
        return false;
      }

      val = (b_vector[static_cast<std::size_t>(local_k)] - sum) / diag;
    }

    MPI_Bcast(&val, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    x_vector[k_index] = val;
  }

  return true;
}

}  // namespace

bool KiselevITestTaskMPI::RunImpl() {
  const auto &[a_in, b_in, band_in] = GetInput();

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int num = 0;
  int band = 0;
  if (rank == 0) {
    num = static_cast<int>(a_in.size());
    band = static_cast<int>(band_in);
  }

  MPI_Bcast(&num, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&band, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int w_coef = (2 * band) + 1;

  std::vector<int> cnt;
  std::vector<int> disp;
  MakePartition(size, num, cnt, disp);

  const int local_rows = cnt[rank];
  const int row0 = disp[rank];

  std::vector<double> a_loc(static_cast<std::size_t>(local_rows) * static_cast<std::size_t>(w_coef), 0.0);
  std::vector<double> b_loc(static_cast<std::size_t>(local_rows), 0.0);

  std::vector<int> cnt_a(size);
  std::vector<int> disp_a(size);
  for (int index = 0; index < size; ++index) {
    cnt_a[index] = cnt[index] * w_coef;
    disp_a[index] = disp[index] * w_coef;
  }

  std::vector<double> send_a;
  std::vector<double> send_b;
  if (rank == 0) {
    send_a.assign(static_cast<std::size_t>(num) * static_cast<std::size_t>(w_coef), 0.0);
    send_b = b_in;

    for (int index = 0; index < num; ++index) {
      for (int j_index = std::max(0, index - band); j_index <= std::min(num - 1, index + band); ++j_index) {
        const auto row_index = static_cast<std::size_t>(index);
        const auto col_index = static_cast<std::size_t>(j_index);
        const auto band_offset = static_cast<std::size_t>(j_index - (index - band));
        const auto linear_index = (row_index * static_cast<std::size_t>(w_coef)) + band_offset;

        send_a[linear_index] = a_in[row_index][col_index];
      }
    }
  }

  MPI_Scatterv(rank == 0 ? send_a.data() : nullptr, cnt_a.data(), disp_a.data(), MPI_DOUBLE, a_loc.data(), cnt_a[rank],
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? send_b.data() : nullptr, cnt.data(), disp.data(), MPI_DOUBLE, b_loc.data(), local_rows,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (!EliminateForward(num, band, w_coef, rank, row0, local_rows, cnt, disp, a_loc, b_loc)) {
    return false;
  }

  std::vector<double> x_vector;
  if (!EliminateBackward(num, band, w_coef, rank, row0, cnt, disp, a_loc, b_loc, x_vector)) {
    return false;
  }

  GetOutput().swap(x_vector);
  return true;
}

bool KiselevITestTaskMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace kiselev_i_gauss_method_horizontal_tape_scheme
