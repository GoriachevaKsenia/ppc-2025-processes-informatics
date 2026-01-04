#include "shvetsova_k_gausse_vert_strip/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "shvetsova_k_gausse_vert_strip/common/include/common.hpp"

namespace shvetsova_k_gausse_vert_strip {

// вспомогательные функции

int ShvetsovaKGaussVertStripMPI::GetOwnerOfColumn(int k, int n, int size) {
  int base = n / size;
  int rem = n % size;
  int border = rem * (base + 1);
  return (k < border) ? k / (base + 1) : rem + ((k - border) / base);
}

int ShvetsovaKGaussVertStripMPI::GetColumnStartIndex(int rank, int n, int size) {
  int base = n / size;
  int rem = n % size;
  return (rank < rem) ? rank * (base + 1) : (rem * (base + 1)) + ((rank - rem) * base);
}

int ShvetsovaKGaussVertStripMPI::GetColumnEndIndex(int rank, int n, int size) {
  return GetColumnStartIndex(rank + 1, n, size);
}

int ShvetsovaKGaussVertStripMPI::CalculateRibWidth(const std::vector<std::vector<double>> &matrix, int n) {
  int rib_width = 1;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i != j && std::abs(matrix[i][j]) > 1e-12) {
        rib_width = std::max(rib_width, std::abs(i - j) + 1);
      }
    }
  }

  return rib_width;
}

void ShvetsovaKGaussVertStripMPI::ForwardElimination(int n, int rank, int size, int c0, int local_cols,
                                                     std::vector<std::vector<double>> &a_local,
                                                     std::vector<double> &b) const {
  for (int k = 0; k < n; ++k) {
    int owner = GetOwnerOfColumn(k, n, size);
    double pivot = 0.0;

    if (rank == owner) {
      pivot = a_local[k][k - c0];
    }
    MPI_Bcast(&pivot, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

    if (std::abs(pivot) < 1e-12) {
      // единое поведение для всех рангов
      throw std::runtime_error("Zero pivot encountered");
    }

    for (int j = 0; j < local_cols; ++j) {
      a_local[k][j] /= pivot;
    }
    b[k] /= pivot;

    for (int i = k + 1; i < std::min(n, k + size_of_rib_); ++i) {
      double factor = 0.0;

      if (rank == owner) {
        factor = a_local[i][k - c0];
      }
      MPI_Bcast(&factor, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

      for (int j = 0; j < local_cols; ++j) {
        a_local[i][j] -= factor * a_local[k][j];
      }
      b[i] -= factor * b[k];
    }
  }
}

std::vector<double> ShvetsovaKGaussVertStripMPI::BackSubstitution(int n, int rank, int size, int c0,
                                                                  const std::vector<std::vector<double>> &a_local,
                                                                  const std::vector<double> &b) const {
  std::vector<double> x(n, 0.0);

  for (int k = n - 1; k >= 0; --k) {
    double sum = b[k];

    for (int j = k + 1; j < std::min(n, k + size_of_rib_); ++j) {
      int owner = GetOwnerOfColumn(j, n, size);
      double val = 0.0;

      if (rank == owner) {
        val = a_local[k][j - c0];
      }
      MPI_Bcast(&val, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

      sum -= val * x[j];
    }

    int owner = GetOwnerOfColumn(k, n, size);
    if (rank == owner) {
      x[k] = sum;
    }
    MPI_Bcast(&x[k], 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
  }

  return x;
}

void ShvetsovaKGaussVertStripMPI::ScatterColumns(int n, int rank, int size, int c0, int local_cols,
                                                 const std::vector<std::vector<double>> &matrix,
                                                 std::vector<std::vector<double>> &a_local) {
  if (rank == 0) {
    // копируем свои столбцы
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < local_cols; ++j) {
        a_local[i][j] = matrix[i][c0 + j];
      }
    }

    // рассылаем остальным
    for (int rr = 1; rr < size; ++rr) {
      int rs = GetColumnStartIndex(rr, n, size);
      int re = GetColumnEndIndex(rr, n, size);
      int cols = re - rs;

      for (int i = 0; i < n; ++i) {
        MPI_Send(&matrix[i][rs], cols, MPI_DOUBLE, rr, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    // принимаем свои столбцы
    for (int i = 0; i < n; ++i) {
      MPI_Recv(a_local[i].data(), local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

// основные функции
ShvetsovaKGaussVertStripMPI::ShvetsovaKGaussVertStripMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool ShvetsovaKGaussVertStripMPI::ValidationImpl() {
  return true;
}

bool ShvetsovaKGaussVertStripMPI::PreProcessingImpl() {
  input_data_ = GetInput();
  return true;
}

bool ShvetsovaKGaussVertStripMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &matrix = input_data_.first;
  const auto &b_in = input_data_.second;

  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(matrix.size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n == 0) {
    GetOutput().clear();
    return true;
  }

  // вычисление размера ленты
  if (rank == 0) {
    size_of_rib_ = CalculateRibWidth(matrix, n);
  }
  MPI_Bcast(&size_of_rib_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int c0 = GetColumnStartIndex(rank, n, size);
  int c1 = GetColumnEndIndex(rank, n, size);
  int local_cols = c1 - c0;

  std::vector<std::vector<double>> a_local(n, std::vector<double>(local_cols));
  std::vector<double> b = b_in;

  // разделяем столбцы
  ScatterColumns(n, rank, size, c0, local_cols, matrix, a_local);

  MPI_Bcast(b.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // прямой ход
  try {
    ForwardElimination(n, rank, size, c0, local_cols, a_local, b);
  } catch (...) {
    GetOutput() = std::vector<double>(n, 0.0);
    return true;
  }

  // обратный ход
  std::vector<double> x = BackSubstitution(n, rank, size, c0, a_local, b);

  MPI_Barrier(MPI_COMM_WORLD);
  GetOutput() = x;
  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

bool ShvetsovaKGaussVertStripMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shvetsova_k_gausse_vert_strip
