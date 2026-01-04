#include "sannikov_i_horizontal_band_gauss/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "sannikov_i_horizontal_band_gauss/common/include/common.hpp"

namespace sannikov_i_horizontal_band_gauss {

SannikovIHorizontalBandGaussMPI::SannikovIHorizontalBandGaussMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &input_buffer = GetInput();
  InType tmp(in);
  input_buffer.swap(tmp);
  GetOutput().clear();
}

bool SannikovIHorizontalBandGaussMPI::ValidationImpl() {
  (void)this;

  const auto &a = std::get<0>(GetInput());
  const auto &rhs = std::get<1>(GetInput());

  if (a.empty() || rhs.empty()) {
    return false;
  }
  if (a.size() != rhs.size()) {
    return false;
  }
  return GetOutput().empty();
}

bool SannikovIHorizontalBandGaussMPI::PreProcessingImpl() {
  (void)this;

  GetOutput().clear();
  return GetOutput().empty();
}
namespace {
void BuildRowPartition(int size, int n, std::vector<int> *counts, std::vector<int> *displs) {
  counts->assign(size, 0);
  displs->assign(size, 0);

  const int base = n / size;
  const int rem = n % size;

  int disp = 0;
  for (int res = 0; res < size; ++res) {
    (*counts)[res] = base + ((res < rem) ? 1 : 0);
    (*displs)[res] = disp;
    disp += (*counts)[res];
  }
}
void FillPivotSegmentIfOwner(int k, int j_end, int band_eff, int w, int rank, int owner, int row_begin,
                             const std::vector<double> &a_loc, const std::vector<double> &b_loc,
                             std::vector<double> *pivot_seg, double *pivot_b) {
  if (rank != owner) {
    return;
  }

  const int loc_k = k - row_begin;
  const double *rowk = &a_loc[static_cast<std::size_t>(loc_k) * static_cast<std::size_t>(w)];

  for (int j = k; j <= j_end; ++j) {
    const int band_off = j - (k - band_eff);
    (*pivot_seg)[static_cast<std::size_t>(j - k)] = rowk[static_cast<std::size_t>(band_off)];
  }

  *pivot_b = b_loc[static_cast<std::size_t>(loc_k)];
}

void EliminateLocalRows(int k, int j_end, int band_eff, int w, int row_begin, int loc_rows, std::vector<double> &a_loc,
                        std::vector<double> &b_loc, const std::vector<double> &pivot_seg, double pivot_b,
                        double pivot) {
  const int i_start = std::max(row_begin, k + 1);
  const int i_end = std::min(row_begin + loc_rows - 1, k + band_eff);

  for (int i = i_start; i <= i_end; ++i) {
    const int loc_i = i - row_begin;
    double *rowi = &a_loc[static_cast<std::size_t>(loc_i) * static_cast<std::size_t>(w)];

    const int off_ik = k - (i - band_eff);
    if (off_ik < 0 || off_ik >= w) {
      continue;
    }

    const double mn = rowi[static_cast<std::size_t>(off_ik)] / pivot;
    rowi[static_cast<std::size_t>(off_ik)] = 0.0;

    for (int j = k + 1; j <= j_end; ++j) {
      const int off_ij = j - (i - band_eff);
      if (off_ij >= 0 && off_ij < w) {
        rowi[static_cast<std::size_t>(off_ij)] -= mn * pivot_seg[static_cast<std::size_t>(j - k)];
      }
    }

    b_loc[static_cast<std::size_t>(loc_i)] -= mn * pivot_b;
  }
}

bool ForwardElimination(int n, int band_eff, int w, int rank, int row_begin, int loc_rows,
                        const std::vector<int> &owner_of_row, std::vector<double> &a_loc, std::vector<double> &b_loc) {
  std::vector<double> pivot_seg;
  pivot_seg.reserve(static_cast<std::size_t>(band_eff) + 1U);

  for (int k = 0; k < n; ++k) {
    const int owner = owner_of_row[static_cast<std::size_t>(k)];
    const int j_end = std::min(n - 1, k + band_eff);
    const int len = (j_end - k) + 1;

    pivot_seg.assign(static_cast<std::size_t>(len), 0.0);
    double pivot_b = 0.0;

    FillPivotSegmentIfOwner(k, j_end, band_eff, w, rank, owner, row_begin, a_loc, b_loc, &pivot_seg, &pivot_b);

    MPI_Bcast(pivot_seg.data(), len, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    MPI_Bcast(&pivot_b, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

    const double pivot = pivot_seg[0];
    if (pivot == 0.0) {
      return false;
    }

    EliminateLocalRows(k, j_end, band_eff, w, row_begin, loc_rows, a_loc, b_loc, pivot_seg, pivot_b, pivot);
  }

  return true;
}
bool BackSubstitution(int n, int band_eff, int w, int rank, int row_begin, const std::vector<int> &owner_of_row,
                      const std::vector<double> &a_loc, const std::vector<double> &b_loc, std::vector<double> *x_out) {
  std::vector<double> x(static_cast<std::size_t>(n), 0.0);

  for (int k = n - 1; k >= 0; --k) {
    const int ow = owner_of_row[static_cast<std::size_t>(k)];
    const int j_end = std::min(n - 1, k + band_eff);

    double xk = 0.0;
    if (rank == ow) {
      const int loc_i = k - row_begin;
      const double *rowk = &a_loc[static_cast<std::size_t>(loc_i) * static_cast<std::size_t>(w)];

      double s = 0.0;
      for (int j = k + 1; j <= j_end; ++j) {
        const int band_off = j - (k - band_eff);
        if (band_off >= 0 && band_off < w) {
          s += rowk[static_cast<std::size_t>(band_off)] * x[static_cast<std::size_t>(j)];
        }
      }

      const int off_kk = k - (k - band_eff);
      const double diag = rowk[static_cast<std::size_t>(off_kk)];
      if (diag == 0.0) {
        return false;
      }

      xk = (b_loc[static_cast<std::size_t>(loc_i)] - s) / diag;
    }

    MPI_Bcast(&xk, 1, MPI_DOUBLE, ow, MPI_COMM_WORLD);
    x[static_cast<std::size_t>(k)] = xk;
  }

  x_out->swap(x);
  return true;
}
}  // namespace

bool SannikovIHorizontalBandGaussMPI::RunImpl() {
  const auto &input = GetInput();
  const auto &a_in = std::get<0>(input);
  const auto &b_in = std::get<1>(input);
  const std::size_t band_in = std::get<2>(input);

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  int band_eff = 0;
  if (rank == 0) {
    n = static_cast<int>(a_in.size());
    band_eff = static_cast<int>(band_in);
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&band_eff, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int w = 0;
  w = (2 * band_eff) + 1;

  std::vector<int> row_cnts;
  std::vector<int> row_disp;
  BuildRowPartition(size, n, &row_cnts, &row_disp);
  int loc_rows = 0;
  int row_begin = 0;
  loc_rows = row_cnts[rank];
  row_begin = row_disp[rank];

  std::vector<int> owner_of_row(static_cast<std::size_t>(n), 0);
  for (int res = 0; res < size; ++res) {
    int begin = 0;
    int end = 0;
    begin = row_disp[res];
    end = begin + row_cnts[res];
    for (int i = begin; i < end; ++i) {
      owner_of_row[static_cast<std::size_t>(i)] = res;
    }
  }

  std::vector<double> send_a;
  std::vector<double> send_b;

  if (rank == 0) {
    send_a.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(w), 0.0);
    send_b.resize(static_cast<std::size_t>(n));
    int j_start = 0;
    int j_end = 0;
    int off = 0;
    for (int i = 0; i < n; ++i) {
      j_start = std::max(0, i - band_eff);
      j_end = std::min(n - 1, i + band_eff);

      for (int j = j_start; j <= j_end; ++j) {
        off = j - (i - band_eff);
        send_a[(static_cast<std::size_t>(i) * static_cast<std::size_t>(w)) + static_cast<std::size_t>(off)] =
            a_in[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      }

      send_b[static_cast<std::size_t>(i)] = b_in[static_cast<std::size_t>(i)];
    }
  }
  std::vector<int> counts_a(size);
  std::vector<int> displs_a(size);
  for (int res = 0; res < size; ++res) {
    counts_a[res] = row_cnts[res] * w;
    displs_a[res] = row_disp[res] * w;
  }

  std::vector<double> a_loc(static_cast<std::size_t>(loc_rows * static_cast<std::size_t>(w)), 0.0);
  std::vector<double> b_loc(static_cast<std::size_t>(loc_rows), 0.0);

  MPI_Scatterv(rank == 0 ? send_a.data() : nullptr, counts_a.data(), displs_a.data(), MPI_DOUBLE, a_loc.data(),
               counts_a[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? send_b.data() : nullptr, row_cnts.data(), row_disp.data(), MPI_DOUBLE, b_loc.data(),
               loc_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (!ForwardElimination(n, band_eff, w, rank, row_begin, loc_rows, owner_of_row, a_loc, b_loc)) {
    return false;
  }

  std::vector<double> x;
  if (!BackSubstitution(n, band_eff, w, rank, row_begin, owner_of_row, a_loc, b_loc, &x)) {
    return false;
  }

  GetOutput().swap(x);
  return !GetOutput().empty();
}
bool SannikovIHorizontalBandGaussMPI::PostProcessingImpl() {
  (void)this;

  return !GetOutput().empty();
}

}  // namespace sannikov_i_horizontal_band_gauss
