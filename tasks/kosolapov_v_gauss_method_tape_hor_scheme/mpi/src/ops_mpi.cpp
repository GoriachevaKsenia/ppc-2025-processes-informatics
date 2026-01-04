#include "kosolapov_v_gauss_method_tape_hor_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "kosolapov_v_gauss_method_tape_hor_scheme/common/include/common.hpp"

namespace kosolapov_v_gauss_method_tape_hor_scheme {

KosolapovVGaussMethodTapeHorSchemeMPI::KosolapovVGaussMethodTapeHorSchemeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = {};
}

bool KosolapovVGaussMethodTapeHorSchemeMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &input = GetInput();
    if (input.matrix.empty()) {
      return false;
    }
    for (size_t i = 0; i < input.matrix.size(); ++i) {
      if (input.matrix[i].size() != input.matrix.size()) {
        return false;
      }
    }
    if (input.r_side.size() != static_cast<size_t>(input.matrix.size())) {
      return false;
    }
  }
  return true;
}

bool KosolapovVGaussMethodTapeHorSchemeMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const auto &input = GetInput();
    GetOutput() = std::vector<double>(input.matrix.size(), 0.0);
  } else {
    GetOutput() = std::vector<double>();
  }
  return true;
}

bool KosolapovVGaussMethodTapeHorSchemeMPI::RunImpl() {
  int processes_count = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &processes_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int rows = 0;
  if (rank == 0) {
    const auto &input = GetInput();
    rows = static_cast<int>(input.matrix.size());
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int rows_per_process = rows / processes_count;
  int remainder = rows % processes_count;

  int start_row = (rank * rows_per_process) + std::min(rank, remainder);
  int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0) - 1;
  int local_rows = end_row - start_row + 1;
  std::vector<std::vector<double>> local_matrix;
  std::vector<double> local_rsd;
  std::vector<int> local_row_indices;
  if (rank == 0) {
    DistributeDataFromRoot(local_matrix, local_rsd, local_row_indices, start_row, local_rows, rows, processes_count,
                           rows_per_process, remainder);
  } else {
    local_matrix.resize(local_rows);
    local_rsd.resize(local_rows);
    local_row_indices.resize(local_rows);
    for (int i = 0; i < local_rows; i++) {
      local_matrix[i].resize(rows);
      MPI_Recv(&local_row_indices[i], 1, MPI_INT, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(local_matrix[i].data(), rows, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&local_rsd[i], 1, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  std::vector<int> col_order(rows);
  for (int i = 0; i < rows; ++i) {
    col_order[i] = i;
  }
  // Прямой ход метода Гаусса
  if (!ForwardElimination(rank, rows, start_row, end_row, local_rows, local_matrix, col_order, local_rsd,
                          local_row_indices)) {
    return false;
  }
  // ОБРАТНЫЙ ХОД МЕТОДА ГАУССА
  std::vector<double> global_solution =
      BackwardSubstitution(rows, start_row, end_row, rank, local_rows, local_matrix, local_rsd);
  std::vector<double> final_solution(rows);
  for (int i = 0; i < rows; i++) {
    final_solution[col_order[i]] = global_solution[i];
  }
  MPI_Bcast(final_solution.data(), rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = final_solution;
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool KosolapovVGaussMethodTapeHorSchemeMPI::PostProcessingImpl() {
  return true;
}

void KosolapovVGaussMethodTapeHorSchemeMPI::DistributeDataFromRoot(std::vector<std::vector<double>> &local_matrix,
                                                                   std::vector<double> &local_rsd,
                                                                   std::vector<int> &local_row_indices, int start,
                                                                   int local_rows, int columns, int processes_count,
                                                                   int rows_per_proc, int remainder) {
  const auto &input = GetInput();
  local_matrix.resize(local_rows);
  local_rsd.resize(local_rows);
  local_row_indices.resize(local_rows);

  for (int i = 0; i < local_rows; i++) {
    local_row_indices[i] = start + i;
    local_matrix[i] = input.matrix[start + i];
    local_rsd[i] = input.r_side[start + i];
  }

  for (int proc = 1; proc < processes_count; proc++) {
    const int proc_start = (proc * rows_per_proc) + std::min(proc, remainder);
    const int proc_end = proc_start + rows_per_proc + (proc < remainder ? 1 : 0);
    const int proc_rows_count = proc_end - proc_start;

    for (int i = 0; i < proc_rows_count; i++) {
      int idx = proc_start + i;
      MPI_Send(&idx, 1, MPI_INT, proc, i, MPI_COMM_WORLD);
      MPI_Send(input.matrix[proc_start + i].data(), columns, MPI_DOUBLE, proc, i, MPI_COMM_WORLD);
      MPI_Send(&input.r_side[proc_start + i], 1, MPI_DOUBLE, proc, i, MPI_COMM_WORLD);
    }
  }
}
bool KosolapovVGaussMethodTapeHorSchemeMPI::ForwardElimination(
    int rank, int rows, int start_row, int end_row, int local_rows, std::vector<std::vector<double>> &local_matrix,
    std::vector<int> &col_order, std::vector<double> &local_rsd, std::vector<int> &local_row_indices) {
  for (int step = 0; step < rows; step++) {
    int owner_process = -1;
    if (step >= start_row && step <= end_row) {
      owner_process = rank;
    }
    int global_owner = 0;
    MPI_Allreduce(&owner_process, &global_owner, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    int pivot_row_local_idx = -1;
    double pivot_value = 0.0;
    int pivot_col = step;
    SelectPivot(rank, owner_process, step, start_row, local_rows, rows, local_matrix, pivot_value, pivot_col,
                pivot_row_local_idx);
    MPI_Bcast(&pivot_col, 1, MPI_INT, global_owner, MPI_COMM_WORLD);
    MPI_Bcast(&pivot_value, 1, MPI_DOUBLE, global_owner, MPI_COMM_WORLD);
    MPI_Bcast(&pivot_row_local_idx, 1, MPI_INT, global_owner, MPI_COMM_WORLD);
    if (std::abs(pivot_value) < 1e-12) {
      GetOutput() = std::vector<double>(rows, 0.0);
      MPI_Barrier(MPI_COMM_WORLD);
      return false;
    }
    SwapColumns(pivot_col, step, local_rows, local_matrix, col_order);
    NormalizePivot(rank, owner_process, pivot_row_local_idx, step, rows, pivot_value, local_matrix, local_rsd);
    std::vector<double> pivot_row(rows - step, 0.0);
    double pivot_rhs = 0.0;
    PriparePivot(rank, owner_process, pivot_row_local_idx, step, rows, pivot_row, pivot_rhs, local_matrix, local_rsd);
    MPI_Bcast(pivot_row.data(), rows - step, MPI_DOUBLE, global_owner, MPI_COMM_WORLD);
    MPI_Bcast(&pivot_rhs, 1, MPI_DOUBLE, global_owner, MPI_COMM_WORLD);
    RowSub(local_rows, step, rows, local_matrix, local_rsd, local_row_indices, pivot_row, pivot_rhs);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  return true;
}
std::vector<double> KosolapovVGaussMethodTapeHorSchemeMPI::BackwardSubstitution(
    int rows, int start_row, int end_row, int rank, int local_rows, std::vector<std::vector<double>> &local_matrix,
    std::vector<double> &local_rsd) {
  std::vector<double> global_solution(rows, 0.0);
  for (int step = rows - 1; step >= 0; step--) {
    int owner_process = -1;
    if (step >= start_row && step <= end_row) {
      owner_process = rank;
    }
    int global_owner = 0;
    MPI_Allreduce(&owner_process, &global_owner, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    double x_value = 0.0;
    if (rank == owner_process) {
      int local_idx = step - start_row;
      if (local_idx >= 0 && local_idx < local_rows) {
        x_value = local_rsd[local_idx];
        for (int col = step + 1; col < rows; col++) {
          x_value -= local_matrix[local_idx][col] * global_solution[col];
        }
      }
    }
    MPI_Bcast(&x_value, 1, MPI_DOUBLE, global_owner, MPI_COMM_WORLD);
    global_solution[step] = x_value;
    MPI_Barrier(MPI_COMM_WORLD);
  }
  return global_solution;
}
void KosolapovVGaussMethodTapeHorSchemeMPI::SelectPivot(int rank, int owner_process, int step, int start_row,
                                                        int local_rows, int rows,
                                                        const std::vector<std::vector<double>> &local_matrix,
                                                        double &pivot_value, int &pivot_col, int &pivot_row_local_idx) {
  if (rank == owner_process) {
    int local_idx = step - start_row;
    if (local_idx >= 0 && local_idx < local_rows) {
      double max_val = 0.0;
      int max_col = step;
      for (int col = step; col < rows; col++) {
        double val = std::abs(local_matrix[local_idx][col]);
        if (val > std::abs(max_val)) {
          max_val = local_matrix[local_idx][col];
          max_col = col;
        }
      }
      pivot_value = max_val;
      pivot_col = max_col;
      pivot_row_local_idx = local_idx;
    }
  }
}
void KosolapovVGaussMethodTapeHorSchemeMPI::RowSub(int local_rows, int step, int rows,
                                                   std::vector<std::vector<double>> &local_matrix,
                                                   std::vector<double> &local_rsd,
                                                   const std::vector<int> &local_row_indices,
                                                   const std::vector<double> &pivot_row, double pivot_rhs) {
  for (int i = 0; i < local_rows; i++) {
    int global_row = local_row_indices[i];
    if (global_row > step) {
      double coef = local_matrix[i][step];
      if (std::abs(coef) > 1e-12) {
        for (int j = step; j < rows; ++j) {
          local_matrix[i][j] -= coef * pivot_row[j - step];
        }
        local_rsd[i] -= coef * pivot_rhs;
      }
    }
  }
}
void KosolapovVGaussMethodTapeHorSchemeMPI::SwapColumns(int pivot_col, int step, int local_rows,
                                                        std::vector<std::vector<double>> &local_matrix,
                                                        std::vector<int> &col_order) {
  if (pivot_col != step) {
    for (int i = 0; i < local_rows; i++) {
      std::swap(local_matrix[i][step], local_matrix[i][pivot_col]);
    }
    std::swap(col_order[step], col_order[pivot_col]);
  }
}
void KosolapovVGaussMethodTapeHorSchemeMPI::NormalizePivot(int rank, int owner_process, int pivot_row_local_idx,
                                                           int step, int rows, double pivot_value,
                                                           std::vector<std::vector<double>> &local_matrix,
                                                           std::vector<double> &local_rsd) {
  if (rank == owner_process && pivot_row_local_idx >= 0) {
    for (int j = step; j < rows; j++) {
      local_matrix[pivot_row_local_idx][j] /= pivot_value;
    }
    local_rsd[pivot_row_local_idx] /= pivot_value;
  }
}
void KosolapovVGaussMethodTapeHorSchemeMPI::PriparePivot(int rank, int owner_process, int pivot_row_local_idx, int step,
                                                         int rows, std::vector<double> &pivot_row, double &pivot_rhs,
                                                         const std::vector<std::vector<double>> &local_matrix,
                                                         const std::vector<double> &local_rsd) {
  if (rank == owner_process && pivot_row_local_idx >= 0) {
    for (int j = step; j < rows; j++) {
      pivot_row[j - step] = local_matrix[pivot_row_local_idx][j];
    }
    pivot_rhs = local_rsd[pivot_row_local_idx];
  }
}
}  // namespace kosolapov_v_gauss_method_tape_hor_scheme
