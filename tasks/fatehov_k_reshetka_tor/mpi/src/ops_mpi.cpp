#include "fatehov_k_reshetka_tor/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <vector>

#include "fatehov_k_reshetka_tor/common/include/common.hpp"

namespace fatehov_k_reshetka_tor {

FatehovKReshetkaTorMPI::FatehovKReshetkaTorMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool FatehovKReshetkaTorMPI::ValidationImpl() {
  auto &data = GetInput();
  size_t rows = std::get<0>(data);
  size_t cols = std::get<1>(data);
  auto &vec = std::get<2>(data);

  return (rows > 0 && rows <= kMaxRows) && (cols > 0 && cols <= kMaxCols) && (rows * cols <= kMaxMatrixSize) &&
         (vec.size() == rows * cols) && (!vec.empty());
}

bool FatehovKReshetkaTorMPI::PreProcessingImpl() {
  return true;
}

namespace {

double ProcessValue(double val) {
  double heavy_val = val;
  for (int k = 0; k < 100; ++k) {
    heavy_val = (std::sin(heavy_val) * std::cos(heavy_val)) + std::exp(std::complex<double>(0, heavy_val).real()) +
                std::sqrt(std::abs(heavy_val) + 1.0);
    if (std::isinf(heavy_val)) {
      heavy_val = val;
    }
  }
  return heavy_val;
}

double FindLocalMax(const std::vector<double> &matrix) {
  if (matrix.empty()) {
    return -1e18;
  }

  double local_max = -1e18;
  for (double val : matrix) {
    double processed_val = ProcessValue(val);
    local_max = std::max(processed_val, local_max);
  }
  return local_max;
}

void CalculateGridDimensions(int world_size, int &grid_rows, int &grid_cols) {
  grid_rows = static_cast<int>(std::sqrt(world_size));
  while (world_size % grid_rows != 0) {
    grid_rows--;
  }
  grid_cols = world_size / grid_rows;

  if (grid_rows == 0) {
    grid_rows = 1;
    grid_cols = world_size;
  }
}

void GetGridCoordinates(int world_rank, int grid_cols, int &row, int &col) {
  row = world_rank / grid_cols;
  col = world_rank % grid_cols;
}

int GetTorNeighborRank(int world_rank, int grid_rows, int grid_cols, int delta_row, int delta_col) {
  int row = 0;
  int col = 0;
  GetGridCoordinates(world_rank, grid_cols, row, col);

  row = (row + delta_row + grid_rows) % grid_rows;
  col = (col + delta_col + grid_cols) % grid_cols;

  return (row * grid_cols) + col;
}

void CalculateLocalBlockSize(int world_rank, int grid_rows, int grid_cols, size_t total_rows, size_t total_cols,
                             size_t &my_rows, size_t &my_cols, size_t &start_row, size_t &start_col) {
  int row = 0;
  int col = 0;
  GetGridCoordinates(world_rank, grid_cols, row, col);

  size_t rows_per_proc = total_rows / grid_rows;
  size_t rem_rows = total_rows % grid_rows;

  auto proc_row = static_cast<size_t>(row);
  auto proc_col = static_cast<size_t>(col);

  start_row = (proc_row * rows_per_proc) + std::min<size_t>(proc_row, rem_rows);
  my_rows = rows_per_proc + (proc_row < rem_rows ? 1 : 0);

  size_t cols_per_proc = total_cols / grid_cols;
  size_t rem_cols = total_cols % grid_cols;

  start_col = (proc_col * cols_per_proc) + std::min<size_t>(proc_col, rem_cols);
  my_cols = cols_per_proc + (proc_col < rem_cols ? 1 : 0);
}

void DistributeMatrixData(int world_rank, int world_size, const std::vector<double> &global_matrix, size_t total_rows,
                          size_t total_cols, int grid_rows, int grid_cols, std::vector<double> &local_matrix) {
  if (world_rank == 0) {
    for (int dest = 0; dest < world_size; ++dest) {
      size_t dest_rows = 0;
      size_t dest_cols = 0;
      size_t start_row = 0;
      size_t start_col = 0;
      CalculateLocalBlockSize(dest, grid_rows, grid_cols, total_rows, total_cols, dest_rows, dest_cols, start_row,
                              start_col);

      std::vector<double> buffer(dest_rows * dest_cols);
      for (size_t i = 0; i < dest_rows; ++i) {
        for (size_t j = 0; j < dest_cols; ++j) {
          size_t global_i = start_row + i;
          size_t global_j = start_col + j;
          buffer[(i * dest_cols) + j] = global_matrix[(global_i * total_cols) + global_j];
        }
      }

      if (dest == 0) {
        local_matrix = buffer;
      } else {
        int buffer_size = static_cast<int>(buffer.size());
        MPI_Send(buffer.data(), buffer_size, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    size_t my_rows = 0;
    size_t my_cols = 0;
    size_t start_row = 0;
    size_t start_col = 0;
    CalculateLocalBlockSize(world_rank, grid_rows, grid_cols, total_rows, total_cols, my_rows, my_cols, start_row,
                            start_col);

    local_matrix.resize(my_rows * my_cols);
    int local_size = static_cast<int>(local_matrix.size());
    MPI_Recv(local_matrix.data(), local_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void TorusAllReduce(double &local_value, int world_rank, int grid_rows, int grid_cols) {
  double current_value = local_value;

  for (int col_offset = 1; col_offset < grid_cols; col_offset *= 2) {
    int left_neighbor = GetTorNeighborRank(world_rank, grid_rows, grid_cols, 0, -col_offset);
    int right_neighbor = GetTorNeighborRank(world_rank, grid_rows, grid_cols, 0, col_offset);

    double received_value = 0.0;
    MPI_Status status;

    MPI_Sendrecv(&current_value, 1, MPI_DOUBLE, right_neighbor, 0, &received_value, 1, MPI_DOUBLE, left_neighbor, 0,
                 MPI_COMM_WORLD, &status);

    current_value = std::max(current_value, received_value);
  }

  for (int row_offset = 1; row_offset < grid_rows; row_offset *= 2) {
    int up_neighbor = GetTorNeighborRank(world_rank, grid_rows, grid_cols, -row_offset, 0);
    int down_neighbor = GetTorNeighborRank(world_rank, grid_rows, grid_cols, row_offset, 0);

    double received_value = 0.0;
    MPI_Status status;

    MPI_Sendrecv(&current_value, 1, MPI_DOUBLE, down_neighbor, 0, &received_value, 1, MPI_DOUBLE, up_neighbor, 0,
                 MPI_COMM_WORLD, &status);

    current_value = std::max(current_value, received_value);
  }

  local_value = current_value;
}

}  // namespace

bool FatehovKReshetkaTorMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  size_t total_rows = 0;
  size_t total_cols = 0;
  std::vector<double> global_matrix;

  if (world_rank == 0) {
    auto &data = GetInput();
    total_rows = std::get<0>(data);
    total_cols = std::get<1>(data);
    global_matrix = std::get<2>(data);
  }

  MPI_Bcast(&total_rows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&total_cols, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  int grid_rows = 0;
  int grid_cols = 0;
  CalculateGridDimensions(world_size, grid_rows, grid_cols);

  if (grid_rows * grid_cols != world_size) {
    if (world_rank == 0) {
      std::cerr << "Error: Cannot create grid with " << world_size << " processes" << '\n';
    }
    return false;
  }

  std::vector<double> local_matrix;
  DistributeMatrixData(world_rank, world_size, global_matrix, total_rows, total_cols, grid_rows, grid_cols,
                       local_matrix);

  double local_max = FindLocalMax(local_matrix);

  TorusAllReduce(local_max, world_rank, grid_rows, grid_cols);

  double global_max = local_max;
  MPI_Bcast(&global_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = global_max;
  return true;
}

bool FatehovKReshetkaTorMPI::PostProcessingImpl() {
  return true;
}

}  // namespace fatehov_k_reshetka_tor
