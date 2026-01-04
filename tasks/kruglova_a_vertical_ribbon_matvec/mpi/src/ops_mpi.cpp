#include "kruglova_a_vertical_ribbon_matvec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "kruglova_a_vertical_ribbon_matvec/common/include/common.hpp"

namespace kruglova_a_vertical_ribbon_matvec {

KruglovaAVerticalRibbMatMPI::KruglovaAVerticalRibbMatMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool KruglovaAVerticalRibbMatMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const int rows = std::get<0>(GetInput());
    const int cols = std::get<1>(GetInput());
    const std::vector<double> &matrix = std::get<2>(GetInput());
    const std::vector<double> &vec = std::get<3>(GetInput());

    if (rows <= 0 || cols <= 0) {
      return false;
    }
    if (matrix.size() != static_cast<size_t>(rows) * static_cast<size_t>(cols)) {
      return false;
    }
    if (vec.size() != static_cast<size_t>(cols)) {
      return false;
    }
  }
  return true;
}

bool KruglovaAVerticalRibbMatMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int rows = 0;
  if (rank == 0) {
    rows = std::get<0>(GetInput());
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput().assign(rows, 0.0);

  return true;
}

namespace {

std::vector<int> ComputeSendCountsAndDispls(int cols, int size, std::vector<int> &displs) {
  std::vector<int> sendcounts(size);
  const int base_cols = cols / size;
  const int rem_cols = cols % size;
  int offset = 0;
  for (int i = 0; i < size; ++i) {
    sendcounts[i] = base_cols + (i < rem_cols ? 1 : 0);
    displs[i] = offset;
    offset += sendcounts[i];
  }
  return sendcounts;
}

void TransposeMatrix(const std::vector<double> &matrix, std::vector<double> &transposed, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      transposed[(static_cast<size_t>(j) * rows) + i] = matrix[(static_cast<size_t>(i) * cols) + j];
    }
  }
}

void LocalMatVecMul(const std::vector<double> &local_matrix, const std::vector<double> &local_b,
                    std::vector<double> &local_res, int rows, int local_cols) {
  for (int j = 0; j < local_cols; ++j) {
    const double b_val = local_b[j];
    for (int i = 0; i < rows; ++i) {
      local_res[i] += local_matrix[(static_cast<size_t>(j) * rows) + i] * b_val;
    }
  }
}

}  // namespace

bool KruglovaAVerticalRibbMatMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows = 0;
  int cols = 0;
  if (rank == 0) {
    rows = std::get<0>(GetInput());
    cols = std::get<1>(GetInput());
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> displs(size);
  std::vector<int> sendcounts = ComputeSendCountsAndDispls(cols, size, displs);

  int local_cols = sendcounts[rank];
  std::vector<double> local_b(local_cols);

  const double *global_b_ptr = (rank == 0) ? std::get<3>(GetInput()).data() : nullptr;
  MPI_Scatterv(global_b_ptr, sendcounts.data(), displs.data(), MPI_DOUBLE, local_b.data(), local_cols, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  std::vector<double> transposed_matrix;
  if (rank == 0) {
    const auto &matrix = std::get<2>(GetInput());
    transposed_matrix.resize(static_cast<size_t>(rows) * cols);
    TransposeMatrix(matrix, transposed_matrix, rows, cols);
  }

  std::vector<int> mat_sendcounts(size);
  std::vector<int> mat_displs(size);
  for (int i = 0; i < size; ++i) {
    mat_sendcounts[i] = sendcounts[i] * rows;
    mat_displs[i] = displs[i] * rows;
  }

  std::vector<double> local_matrix(static_cast<size_t>(local_cols) * rows);
  MPI_Scatterv(rank == 0 ? transposed_matrix.data() : nullptr, mat_sendcounts.data(), mat_displs.data(), MPI_DOUBLE,
               local_matrix.data(), local_cols * rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_res(rows, 0.0);
  LocalMatVecMul(local_matrix, local_b, local_res, rows, local_cols);

  double *global_res_ptr = (rank == 0) ? GetOutput().data() : nullptr;
  MPI_Reduce(local_res.data(), global_res_ptr, rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return true;
}

bool KruglovaAVerticalRibbMatMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kruglova_a_vertical_ribbon_matvec
