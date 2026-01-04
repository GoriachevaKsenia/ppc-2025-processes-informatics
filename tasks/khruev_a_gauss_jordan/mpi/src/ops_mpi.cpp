#include "khruev_a_gauss_jordan/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "khruev_a_gauss_jordan/common/include/common.hpp"

namespace khruev_a_gauss_jordan {

int KhruevAGaussJordanMPI::GetGlobalIdx(int local_k, int rank, int size) const {
  int rows = n_ / size;
  int rem = n_ % size;
  if (rank < rem) {
    return (rank * (rows + 1)) + local_k;
  }
  return (rem * (rows + 1)) + ((rank - rem) * rows) + local_k;
}

KhruevAGaussJordanMPI::RowPos KhruevAGaussJordanMPI::GetRowOwner(int global_i, int size) const {
  int rows = n_ / size;
  int rem = n_ % size;

  if (global_i < rem * (rows + 1)) {
    return RowPos{.rank = global_i / (rows + 1), .local_idx = global_i % (rows + 1)};
  }

  int base = rem * (rows + 1);
  return RowPos{.rank = rem + ((global_i - base) / rows), .local_idx = (global_i - base) % rows};
}

int KhruevAGaussJordanMPI::FindLocalPivotIdx(int col, int rank, int size) const {
  int rows = n_ / size;
  int rem = n_ % size;
  int my_rows = (rank < rem) ? (rows + 1) : rows;

  int best = -1;
  double best_val = -1.0;

  for (int k = 0; k < my_rows; ++k) {
    if (GetGlobalIdx(k, rank, size) >= col) {
      double v = std::fabs(local_data_[(k * m_) + col]);
      if (v > best_val) {
        best_val = v;
        best = k;
      }
    }
  }
  return best;
}

KhruevAGaussJordanMPI::PivotPos KhruevAGaussJordanMPI::FindPivot(int col, int rank, int size) {
  PivotPos local{.val = -1.0, .rank = rank};

  int rows = n_ / size;
  int rem = n_ % size;
  int my_rows = (rank < rem) ? (rows + 1) : rows;

  for (int k = 0; k < my_rows; ++k) {
    if (GetGlobalIdx(k, rank, size) >= col) {
      double v = std::fabs(local_data_[(k * m_) + col]);
      local.val = std::max(local.val, v);
    }
  }

  PivotPos global{};
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
  return global;
}

void KhruevAGaussJordanMPI::SwapLocalRows(int a, int b) {
  if (a == b) {
    return;
  }
  auto off_a = static_cast<std::ptrdiff_t>(a) * m_;
  auto off_b = static_cast<std::ptrdiff_t>(b) * m_;
  std::swap_ranges(local_data_.begin() + off_a, local_data_.begin() + off_a + m_, local_data_.begin() + off_b);
}

void KhruevAGaussJordanMPI::SwapRemoteRows(int my_idx, int other_rank) {
  std::vector<double> tmp(m_);
  auto off = static_cast<std::ptrdiff_t>(my_idx) * m_;

  MPI_Sendrecv(local_data_.data() + off, m_, MPI_DOUBLE, other_rank, 0, tmp.data(), m_, MPI_DOUBLE, other_rank, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::ranges::copy(tmp, local_data_.begin() + off);
}

void KhruevAGaussJordanMPI::NormalizePivotRow(int i, const RowPos &pivot, std::vector<double> &pivot_row, int rank) {
  if (rank != pivot.rank) {
    return;
  }

  auto off = static_cast<std::ptrdiff_t>(pivot.local_idx) * m_;
  double *row = local_data_.data() + off;
  double div = row[i];

  if (std::fabs(div) > kEps) {
    for (int j = i; j < m_; ++j) {
      row[j] /= div;
    }
  }
  std::ranges::copy(row, row + m_, pivot_row.begin());
}

void KhruevAGaussJordanMPI::ApplyElimination(int i, const std::vector<double> &pivot_row, int rank, int size) {
  int rows = n_ / size;
  int rem = n_ % size;
  int my_rows = (rank < rem) ? (rows + 1) : rows;

  for (int k = 0; k < my_rows; ++k) {
    if (GetGlobalIdx(k, rank, size) == i) {
      continue;
    }

    auto off = static_cast<std::ptrdiff_t>(k) * m_;
    double *row = local_data_.data() + off;
    double factor = row[i];

    if (std::fabs(factor) > kEps) {
      for (int j = i; j < m_; ++j) {
        row[j] -= factor * pivot_row[j];
      }
    }
  }
}

void KhruevAGaussJordanMPI::SwapRows(int i, int pivot_rank, int rank, int size) {
  RowPos target = GetRowOwner(i, size);

  int local_pivot = (rank == pivot_rank) ? FindLocalPivotIdx(i, rank, size) : -1;

  if (pivot_rank == target.rank) {
    if ((rank == target.rank) && (local_pivot != target.local_idx)) {
      SwapLocalRows(local_pivot, target.local_idx);
    }
    return;
  }

  if (rank == pivot_rank) {
    SwapRemoteRows(local_pivot, target.rank);
  } else if (rank == target.rank) {
    SwapRemoteRows(target.local_idx, pivot_rank);
  }
}

void KhruevAGaussJordanMPI::Eliminate(int i, int rank, int size) {
  RowPos pivot = GetRowOwner(i, size);
  std::vector<double> pivot_row(m_);

  NormalizePivotRow(i, pivot, pivot_row, rank);
  MPI_Bcast(pivot_row.data(), m_, MPI_DOUBLE, pivot.rank, MPI_COMM_WORLD);
  ApplyElimination(i, pivot_row, rank, size);
}

KhruevAGaussJordanMPI::KhruevAGaussJordanMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }
}

bool KhruevAGaussJordanMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int ok = 1;
  if (rank == 0) {
    ok = (!GetInput().empty() && !GetInput()[0].empty()) ? 1 : 0;
  }
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return ok == 1;
}

bool KhruevAGaussJordanMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    n_ = static_cast<int>(GetInput().size());
    m_ = static_cast<int>(GetInput()[0].size());
  }
  MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int rows = n_ / size;
  int rem = n_ % size;
  int my_rows = (rank < rem) ? (rows + 1) : rows;

  local_data_.assign(static_cast<size_t>(my_rows) * m_, 0.0);

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);
  int offset = 0;
  for (int i = 0; i < size; ++i) {
    sendcounts[i] = ((i < rem) ? (rows + 1) : rows) * m_;
    displs[i] = offset;
    offset += sendcounts[i];
  }

  std::vector<double> flat;
  if (rank == 0) {
    flat.resize(static_cast<size_t>(n_) * m_);
    for (int i = 0; i < n_; ++i) {
      std::ranges::copy(GetInput()[i], flat.begin() + static_cast<std::ptrdiff_t>(i) * m_);
    }
  }

  MPI_Scatterv(flat.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, local_data_.data(), my_rows * m_, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
  return true;
}

bool KhruevAGaussJordanMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  for (int i = 0; i < n_; ++i) {
    PivotPos piv = FindPivot(i, rank, size);
    if (piv.val < kEps) {
      continue;
    }
    SwapRows(i, piv.rank, rank, size);
    Eliminate(i, rank, size);
  }
  return true;
}

bool KhruevAGaussJordanMPI::PostProcessingImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows = n_ / size;
  int rem = n_ % size;
  int my_rows = (rank < rem) ? (rows + 1) : rows;

  std::vector<double> full;
  if (rank == 0) {
    full.resize(static_cast<size_t>(n_) * m_);
  }

  std::vector<int> recvcounts(size);
  std::vector<int> displs(size);
  int offset = 0;
  for (int i = 0; i < size; ++i) {
    recvcounts[i] = ((i < rem) ? (rows + 1) : rows) * m_;
    displs[i] = offset;
    offset += recvcounts[i];
  }

  MPI_Gatherv(local_data_.data(), my_rows * m_, MPI_DOUBLE, full.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);

  OutType out(n_);
  if (rank == 0) {
    for (int i = 0; i < n_; ++i) {
      out[i] = full[(i * m_) + (m_ - 1)];
    }
  }

  MPI_Bcast(out.data(), n_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = out;
  return true;
}

}  // namespace khruev_a_gauss_jordan
