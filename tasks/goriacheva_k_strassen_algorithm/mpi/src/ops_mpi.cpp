#include "goriacheva_k_strassen_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "goriacheva_k_strassen_algorithm/common/include/common.hpp"

namespace goriacheva_k_strassen_algorithm {

namespace {

struct Blocks {
  Matrix a11;
  Matrix a12;
  Matrix a21;
  Matrix a22;
  Matrix b11;
  Matrix b12;
  Matrix b21;
  Matrix b22;
};

Blocks SplitMatrices(const Matrix &a, const Matrix &b, std::size_t k) {
  Blocks blk{.a11 = Matrix(k, std::vector<double>(k)),
             .a12 = Matrix(k, std::vector<double>(k)),
             .a21 = Matrix(k, std::vector<double>(k)),
             .a22 = Matrix(k, std::vector<double>(k)),
             .b11 = Matrix(k, std::vector<double>(k)),
             .b12 = Matrix(k, std::vector<double>(k)),
             .b21 = Matrix(k, std::vector<double>(k)),
             .b22 = Matrix(k, std::vector<double>(k))};

  for (std::size_t i = 0; i < k; ++i) {
    for (std::size_t j = 0; j < k; ++j) {
      blk.a11[i][j] = a[i][j];
      blk.a12[i][j] = a[i][j + k];
      blk.a21[i][j] = a[i + k][j];
      blk.a22[i][j] = a[i + k][j + k];

      blk.b11[i][j] = b[i][j];
      blk.b12[i][j] = b[i][j + k];
      blk.b21[i][j] = b[i + k][j];
      blk.b22[i][j] = b[i + k][j + k];
    }
  }

  return blk;
}

Matrix ComputeMi(int task_id, const Blocks &blk) {
  switch (task_id) {
    case 0:
      return Strassen(Add(blk.a11, blk.a22), Add(blk.b11, blk.b22));
    case 1:
      return Strassen(Add(blk.a21, blk.a22), blk.b11);
    case 2:
      return Strassen(blk.a11, Sub(blk.b12, blk.b22));
    case 3:
      return Strassen(blk.a22, Sub(blk.b21, blk.b11));
    case 4:
      return Strassen(Add(blk.a11, blk.a12), blk.b22);
    case 5:
      return Strassen(Sub(blk.a21, blk.a11), Add(blk.b11, blk.b12));
    case 6:
      return Strassen(Sub(blk.a12, blk.a22), Add(blk.b21, blk.b22));
    default:
      return {};
  }
}

void ComputeMissingTasks(std::vector<Matrix> &m, int start_task, const Blocks &blk) {
  for (int task = start_task; task < 7; ++task) {
    m[task] = ComputeMi(task, blk);
  }
}

Matrix AssembleResult(const std::vector<Matrix> &m, std::size_t k) {
  std::size_t n = 2 * k;
  Matrix c(n, std::vector<double>(n));

  for (std::size_t i = 0; i < k; ++i) {
    for (std::size_t j = 0; j < k; ++j) {
      c[i][j] = m[0][i][j] + m[3][i][j] - m[4][i][j] + m[6][i][j];
      c[i][j + k] = m[2][i][j] + m[4][i][j];
      c[i + k][j] = m[1][i][j] + m[3][i][j];
      c[i + k][j + k] = m[0][i][j] - m[1][i][j] + m[2][i][j] + m[5][i][j];
    }
  }

  return c;
}

}  // namespace

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
  auto blocks = SplitMatrices(a, b, k);

  int num_tasks = std::min(7, size);
  int task_id = rank % num_tasks;

  MPI_Comm subcomm = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, task_id, rank, &subcomm);

  int sub_rank = 0;
  MPI_Comm_rank(subcomm, &sub_rank);

  Matrix mi;
  if (sub_rank == 0) {
    mi = ComputeMi(task_id, blocks);
  }

  MPI_Comm_free(&subcomm);

  std::vector<Matrix> m(7);

  if (rank == 0) {
    if (task_id < num_tasks && sub_rank == 0) {
      m[task_id] = mi;
    }

    for (int i = 1; i < num_tasks; ++i) {
      int tid = 0;
      std::vector<double> buf(k * k);
      MPI_Status status;

      MPI_Recv(&tid, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(buf.data(), static_cast<int>(k * k), MPI_DOUBLE, status.MPI_SOURCE, 1, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      m[tid] = UnFlatten(buf, k);
    }

    ComputeMissingTasks(m, num_tasks, blocks);
  } else if (sub_rank == 0) {
    auto buf = Flatten(mi);
    MPI_Send(&task_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(buf.data(), static_cast<int>(k * k), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
  }

  Matrix c;
  std::vector<double> flat_c;

  if (rank == 0) {
    c = AssembleResult(m, k);
    flat_c = Flatten(c);
  } else {
    flat_c.resize(n * n);
  }

  MPI_Bcast(flat_c.data(), static_cast<int>(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    c = UnFlatten(flat_c, n);
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
