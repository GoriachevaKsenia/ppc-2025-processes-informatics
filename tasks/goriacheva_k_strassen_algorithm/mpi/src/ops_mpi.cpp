#include "goriacheva_k_strassen_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "goriacheva_k_strassen_algorithm/common/include/common.hpp"

namespace goriacheva_k_strassen_algorithm {

GoriachevaKStrassenAlgorithmMPI::GoriachevaKStrassenAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool GoriachevaKStrassenAlgorithmMPI::ValidationImpl() {
  return IsSquare(GetInput().a) && IsSquare(GetInput().b) && GetInput().a.size() == GetInput().b.size();
}

bool GoriachevaKStrassenAlgorithmMPI::PreProcessingImpl() {
  input_matrices_ = GetInput();
  return true;
}

Matrix GoriachevaKStrassenAlgorithmMPI::MpiStrassenTop(const Matrix &a, const Matrix &b) {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::size_t n = a.size();
  if (size == 1 || n <= 1) {
    return Strassen(a, b);
  }

  std::size_t k = n / 2;

  Matrix a11(k, std::vector<double>(k));
  Matrix a12(k, std::vector<double>(k));
  Matrix a21(k, std::vector<double>(k));
  Matrix a22(k, std::vector<double>(k));
  Matrix b11(k, std::vector<double>(k));
  Matrix b12(k, std::vector<double>(k));
  Matrix b21(k, std::vector<double>(k));
  Matrix b22(k, std::vector<double>(k));

  for (std::size_t i = 0; i < k; ++i) {
    for (std::size_t j = 0; j < k; ++j) {
      a11[i][j] = a[i][j];
      a12[i][j] = a[i][j + k];
      a21[i][j] = a[i + k][j];
      a22[i][j] = a[i + k][j + k];
      b11[i][j] = b[i][j];
      b12[i][j] = b[i][j + k];
      b21[i][j] = b[i + k][j];
      b22[i][j] = b[i + k][j + k];
    }
  }

  int num_tasks = std::min(7, size);
  int task_id = rank % num_tasks;

  MPI_Comm Subcomm;
  MPI_Comm_split(MPI_COMM_WORLD, task_id, rank, &Subcomm);

  int Sub_rank;
  MPI_Comm_rank(Subcomm, &Sub_rank);

  Matrix mi;
  if (Sub_rank == 0) {
    switch (task_id) {
      case 0:
        mi = Strassen(Add(a11, a22), Add(b11, b22));
        break;
      case 1:
        mi = Strassen(Add(a21, a22), b11);
        break;
      case 2:
        mi = Strassen(a11, Sub(b12, b22));
        break;
      case 3:
        mi = Strassen(a22, Sub(b21, b11));
        break;
      case 4:
        mi = Strassen(Add(a11, a12), b22);
        break;
      case 5:
        mi = Strassen(Sub(a21, a11), Add(b11, b12));
        break;
      case 6:
        mi = Strassen(Sub(a12, a22), Add(b21, b22));
        break;
    }
  }

  MPI_Comm_free(&Subcomm);

  std::vector<Matrix> m(7);

  if (rank == 0) {
    if (task_id < num_tasks && Sub_rank == 0) {
      m[task_id] = mi;
    }

    for (int i = 1; i < num_tasks; ++i) {
      int tid;
      std::vector<double> buf(k * k);
      MPI_Status status;

      MPI_Recv(&tid, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(buf.data(), k * k, MPI_DOUBLE, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      m[tid] = UnFlatten(buf, k);
    }

    for (int t = num_tasks; t < 7; ++t) {
      switch (t) {
        case 0:
          m[0] = Strassen(Add(a11, a22), Add(b11, b22));
          break;
        case 1:
          m[1] = Strassen(Add(a21, a22), b11);
          break;
        case 2:
          m[2] = Strassen(a11, Sub(b12, b22));
          break;
        case 3:
          m[3] = Strassen(a22, Sub(b21, b11));
          break;
        case 4:
          m[4] = Strassen(Add(a11, a12), b22);
          break;
        case 5:
          m[5] = Strassen(Sub(a21, a11), Add(b11, b12));
          break;
        case 6:
          m[6] = Strassen(Sub(a12, a22), Add(b21, b22));
          break;
      }
    }
  } else if (Sub_rank == 0) {
    auto buf = Flatten(mi);
    MPI_Send(&task_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(buf.data(), k * k, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
  }

  Matrix c(n, std::vector<double>(n));
  if (rank == 0) {
    for (std::size_t i = 0; i < k; ++i) {
      for (std::size_t j = 0; j < k; ++j) {
        c[i][j] = m[0][i][j] + m[3][i][j] - m[4][i][j] + m[6][i][j];
        c[i][j + k] = m[2][i][j] + m[4][i][j];
        c[i + k][j] = m[1][i][j] + m[3][i][j];
        c[i + k][j + k] = m[0][i][j] - m[1][i][j] + m[2][i][j] + m[5][i][j];
      }
    }
  }

  auto flatC = Flatten(c);
  MPI_Bcast(flatC.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    c = UnFlatten(flatC, n);
  }

  return c;
}

bool GoriachevaKStrassenAlgorithmMPI::RunImpl() {
  const auto &a = input_matrices_.a;
  const auto &b = input_matrices_.b;

  std::size_t n = a.size();
  std::size_t m = NextPowerOfTwo(n);

  Matrix a_pad = (n == m) ? a : PadMatrix(a, m);
  Matrix b_pad = (n == m) ? b : PadMatrix(b, m);

  Matrix c_pad = MpiStrassenTop(a_pad, b_pad);
  result_matrix_ = (n == m) ? c_pad : CropMatrix(c_pad, n);

  return true;
}

bool GoriachevaKStrassenAlgorithmMPI::PostProcessingImpl() {
  GetOutput() = result_matrix_;
  return true;
}

}  // namespace goriacheva_k_strassen_algorithm
