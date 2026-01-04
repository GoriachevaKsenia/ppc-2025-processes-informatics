#include "liulin_y_vert_strip_diag_matrix_vect_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "liulin_y_vert_strip_diag_matrix_vect_mult/common/include/common.hpp"

namespace liulin_y_vert_strip_diag_matrix_vect_mult {

LiulinYVertStripDiagMatrixVectMultMPI::LiulinYVertStripDiagMatrixVectMultMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  auto &matrix = std::get<0>(GetInput());
  auto &vect = std::get<1>(GetInput());

  const auto &input_matrix = std::get<0>(in);
  const auto &input_vect = std::get<1>(in);

  matrix.clear();
  vect.clear();

  if (!input_matrix.empty()) {
    matrix = input_matrix;
  }

  if (!input_vect.empty()) {
    vect = input_vect;
  }

  GetOutput().clear();
}

bool LiulinYVertStripDiagMatrixVectMultMPI::ValidationImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank != 0) {
    return true;
  }
  return true;
}
bool LiulinYVertStripDiagMatrixVectMultMPI::PreProcessingImpl() {
  return true;
}

namespace {
void PrepareCounts(int rows, int cols, int world_size, std::vector<int> &sendcounts, std::vector<int> &displs) {
  sendcounts.assign(world_size, 0);
  displs.assign(world_size, 0);

  int base_cols = cols / world_size;
  int remainder = cols % world_size;

  for (int proc = 0; proc < world_size; ++proc) {
    int local_cols = base_cols + (proc < remainder ? 1 : 0);
    sendcounts[proc] = local_cols * rows;
    if (proc > 0) {
      displs[proc] = displs[proc - 1] + sendcounts[proc - 1];
    }
  }
}

void FillFlatMatrix(const std::vector<std::vector<int>> &matrix, int rows, int cols, std::vector<int> &flat_matrix) {
  flat_matrix.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
  for (int col = 0; col < cols; ++col) {
    for (int row = 0; row < rows; ++row) {
      flat_matrix[(col * rows) + row] = matrix[row][col];
    }
  }
}

std::vector<int> ComputeLocalPartialRes(const std::vector<int> &local_data, const std::vector<int> &local_vect,
                                        int rows, int local_cols) {
  std::vector<int> local_partial(rows, 0);
  for (int col_idx = 0; col_idx < local_cols; ++col_idx) {
    for (int row = 0; row < rows; ++row) {
      local_partial[row] += local_data[(col_idx * rows) + row] * local_vect[col_idx];
    }
  }
  return local_partial;
}
}  // namespace

bool LiulinYVertStripDiagMatrixVectMultMPI::RunImpl() {
  const auto &input = GetInput();
  auto &out = GetOutput();

  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int rows = 0;
  int cols = 0;
  if (world_rank == 0) {
    const auto &matrix = std::get<0>(input);
    if (!matrix.empty() && !matrix[0].empty()) {
      rows = static_cast<int>(matrix.size());
      cols = static_cast<int>(matrix[0].size());
    }
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rows <= 0 || cols <= 0) {
    out.clear();
    return true;
  }

  out.assign(rows, 0);

  std::vector<int> sendcounts;
  std::vector<int> displs;
  PrepareCounts(rows, cols, world_size, sendcounts, displs);

  std::vector<int> recvcounts(world_size);
  std::vector<int> displs_gather(world_size, 0);
  for (int proc = 0; proc < world_size; ++proc) {
    recvcounts[proc] = sendcounts[proc] / rows;
    if (proc > 0) {
      displs_gather[proc] = displs_gather[proc - 1] + recvcounts[proc - 1];
    }
  }

  int local_cols = recvcounts[world_rank];
  int local_elements = local_cols * rows;
  std::vector<int> local_data(local_elements);

  std::vector<int> flat_matrix;
  if (world_rank == 0) {
    const auto &matrix = std::get<0>(input);
    FillFlatMatrix(matrix, rows, cols, flat_matrix);
  }

  MPI_Scatterv(world_rank == 0 ? flat_matrix.data() : nullptr, sendcounts.data(), displs.data(), MPI_INT,
               local_data.data(), local_elements, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_vect(local_cols);
  if (world_rank == 0) {
    const auto &vect = std::get<1>(input);
    MPI_Scatterv(vect.data(), recvcounts.data(), displs_gather.data(), MPI_INT, local_vect.data(), local_cols, MPI_INT,
                 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, recvcounts.data(), displs_gather.data(), MPI_INT, local_vect.data(), local_cols, MPI_INT, 0,
                 MPI_COMM_WORLD);
  }

  std::vector<int> local_partial = ComputeLocalPartialRes(local_data, local_vect, rows, local_cols);

  MPI_Reduce(local_partial.data(), out.data(), rows, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(out.data(), rows, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool LiulinYVertStripDiagMatrixVectMultMPI::PostProcessingImpl() {
  return true;
}

}  // namespace liulin_y_vert_strip_diag_matrix_vect_mult
