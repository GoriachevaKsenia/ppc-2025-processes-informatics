#include "gauss_jordan/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "gauss_jordan/common/include/common.hpp"

namespace gauss_jordan {

namespace {

constexpr double kEpsilon = 1e-12;

inline bool IsNumericallyZero(double value) {
  return std::fabs(value) < kEpsilon;
}

bool IsZeroRow(const std::vector<double> &row, int columns_minus_one) {
  for (int j = 0; j < columns_minus_one; ++j) {
    if (!IsNumericallyZero(row[j])) {
      return false;
    }
  }
  return true;
}

bool ValidateMatrixDimensions(const std::vector<std::vector<double>> &augmented_matrix, int equations_count,
                              int augmented_columns, int rank) {
  if (rank == 0) {
    for (int i = 1; i < equations_count; ++i) {
      if (augmented_matrix[i].size() != static_cast<size_t>(augmented_columns)) {
        return false;
      }
    }
  }
  return true;
}

bool CheckIfZeroMatrix(const std::vector<std::vector<double>> &augmented_matrix, int equations_count,
                       int augmented_columns, int rank) {
  if (rank == 0) {
    for (int i = 0; i < equations_count; ++i) {
      for (int j = 0; j < augmented_columns; ++j) {
        if (std::abs(augmented_matrix[i][j]) > 1e-12) {
          return false;
        }
      }
    }
  }
  return true;
}

void HandleRankZeroOutput(int rank, std::vector<double> &output, std::vector<double> result) {
  if (rank == 0) {
    output = std::move(result);
  }
}

bool ShouldReturnEarly(int rank, const std::vector<std::vector<double>> &matrix, int equations_count,
                       int augmented_columns, std::vector<double> &output) {  // Добавлен параметр output
  if (augmented_columns == 0) {
    HandleRankZeroOutput(rank, output, std::vector<double>());
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  bool valid_matrix = ValidateMatrixDimensions(matrix, equations_count, augmented_columns, rank);
  MPI_Bcast(&valid_matrix, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  if (!valid_matrix) {
    HandleRankZeroOutput(rank, output, std::vector<double>());
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  if (augmented_columns <= 1) {
    HandleRankZeroOutput(rank, output, std::vector<double>());
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  bool is_zero_matrix = CheckIfZeroMatrix(matrix, equations_count, augmented_columns, rank);
  MPI_Bcast(&is_zero_matrix, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  if (is_zero_matrix) {
    HandleRankZeroOutput(rank, output, std::vector<double>());
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  return false;
}

bool HasValidRank(int global_rank, int augmented_columns, int equations_count) {
  return global_rank >= augmented_columns - 1 && global_rank >= equations_count;
}

void ExchangeRows(std::vector<std::vector<double>> &augmented_matrix, int first_row, int second_row, int columns) {
  if (first_row == second_row) {
    return;
  }

  for (int col_idx = 0; col_idx < columns; ++col_idx) {
    std::swap(augmented_matrix[first_row][col_idx], augmented_matrix[second_row][col_idx]);
  }
}

void NormalizeRow(std::vector<std::vector<double>> &augmented_matrix, int row_index, double normalizer, int columns) {
  if (IsNumericallyZero(normalizer)) {
    return;
  }

  for (int col_idx = 0; col_idx < columns; ++col_idx) {
    augmented_matrix[row_index][col_idx] /= normalizer;
  }
}

int LocatePivotIndex(const std::vector<std::vector<double>> &augmented_matrix, int start_row, int column,
                     int total_rows) {
  for (int row_idx = start_row; row_idx < total_rows; ++row_idx) {
    if (!IsNumericallyZero(augmented_matrix[row_idx][column])) {
      return row_idx;
    }
  }
  return -1;
}

void EliminateFromRow(std::vector<std::vector<double>> &augmented_matrix, int target_row, int source_row,
                      double elimination_coefficient, int columns) {
  if (IsNumericallyZero(elimination_coefficient)) {
    return;
  }

  for (int col_idx = 0; col_idx < columns; ++col_idx) {
    augmented_matrix[target_row][col_idx] -= elimination_coefficient * augmented_matrix[source_row][col_idx];
  }
}

void BroadcastRow(std::vector<std::vector<double>> &augmented_matrix, int row, int columns) {
  MPI_Bcast(augmented_matrix[row].data(), columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void BroadcastMatrix(std::vector<std::vector<double>> &augmented_matrix, int rows, int columns) {
  for (int i = 0; i < rows; ++i) {
    MPI_Bcast(augmented_matrix[i].data(), columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}

int FindPivotMPI(const std::vector<std::vector<double>> &augmented_matrix, int current_row, int current_col,
                 int equations_count) {
  return LocatePivotIndex(augmented_matrix, current_row, current_col, equations_count);
}

void ProcessPivotRow(std::vector<std::vector<double>> &augmented_matrix, int current_row, int current_col,
                     int pivot_row, int augmented_columns) {
  if (pivot_row != current_row) {
    ExchangeRows(augmented_matrix, current_row, pivot_row, augmented_columns);
  }

  double pivot_value = augmented_matrix[current_row][current_col];
  if (!IsNumericallyZero(pivot_value)) {
    NormalizeRow(augmented_matrix, current_row, pivot_value, augmented_columns);
  }

  BroadcastRow(augmented_matrix, current_row, augmented_columns);
}

void EliminateColumnFromOtherRows(std::vector<std::vector<double>> &augmented_matrix, int current_row, int current_col,
                                  int equations_count, int augmented_columns) {
  for (int row_idx = 0; row_idx < equations_count; ++row_idx) {
    if (row_idx == current_row) {
      continue;
    }

    double coefficient = augmented_matrix[row_idx][current_col];
    if (!IsNumericallyZero(coefficient)) {
      EliminateFromRow(augmented_matrix, row_idx, current_row, coefficient, augmented_columns);
    }
  }
}

void TransformToReducedRowEchelonFormMPI(std::vector<std::vector<double>> &augmented_matrix, int equations_count,
                                         int augmented_columns) {
  int current_row = 0;
  int current_col = 0;

  while (current_row < equations_count && current_col < augmented_columns - 1) {
    int pivot_row = FindPivotMPI(augmented_matrix, current_row, current_col, equations_count);

    if (pivot_row == -1) {
      current_col++;
      continue;
    }

    ProcessPivotRow(augmented_matrix, current_row, current_col, pivot_row, augmented_columns);

    EliminateColumnFromOtherRows(augmented_matrix, current_row, current_col, equations_count, augmented_columns);

    BroadcastMatrix(augmented_matrix, equations_count, augmented_columns);
    MPI_Barrier(MPI_COMM_WORLD);

    current_row++;
    current_col++;
  }
}

bool HasInconsistentEquation(const std::vector<std::vector<double>> &augmented_matrix, int equations_count,
                             int augmented_columns) {
  for (int row_idx = 0; row_idx < equations_count; ++row_idx) {
    if (IsZeroRow(augmented_matrix[row_idx], augmented_columns - 1) &&
        !IsNumericallyZero(augmented_matrix[row_idx][augmented_columns - 1])) {
      return true;
    }
  }
  return false;
}

int ComputeMatrixRank(const std::vector<std::vector<double>> &augmented_matrix, int equations_count,
                      int augmented_columns) {
  int rank = 0;
  for (int row_idx = 0; row_idx < equations_count; ++row_idx) {
    if (!IsZeroRow(augmented_matrix[row_idx], augmented_columns - 1)) {
      ++rank;
    }
  }
  return rank;
}

std::vector<double> ComputeSolutionVector(const std::vector<std::vector<double>> &matrix, int n, int m) {
  std::vector<double> solution(m - 1, 0.0);
  int vars = m - 1;
  int rows_to_process = (n < vars) ? n : vars;

  for (int i = 0; i < rows_to_process; ++i) {
    int pivot_col = -1;

    for (int j = 0; j < vars; ++j) {
      if (std::abs(matrix[i][j]) > 1e-10) {
        pivot_col = j;
        break;
      }
    }

    if (pivot_col >= 0) {
      solution[pivot_col] = matrix[i][vars];
    } else {
      solution[i] = 0.0;
    }
  }

  return solution;
}

void ProcessReducedMatrix(const std::vector<std::vector<double>> &augmented_matrix, int equations_count,
                          int augmented_columns, bool &global_inconsistent, int &global_rank,
                          std::vector<double> &local_solution, int /*rank*/) {
  bool local_inconsistent = HasInconsistentEquation(augmented_matrix, equations_count, augmented_columns);
  int local_rank = ComputeMatrixRank(augmented_matrix, equations_count, augmented_columns);

  if (!local_inconsistent && local_rank >= augmented_columns - 1 && local_rank >= equations_count) {
    local_solution = ComputeSolutionVector(augmented_matrix, equations_count, augmented_columns);
  }

  MPI_Reduce(&local_inconsistent, &global_inconsistent, 1, MPI_C_BOOL, MPI_LOR, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_rank, &global_rank, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Bcast(&global_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_inconsistent, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
}

void BroadcastSolution(std::vector<double> &solution, int rank) {
  int solution_size = static_cast<int>(solution.size());
  MPI_Bcast(&solution_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    solution.resize(static_cast<size_t>(solution_size));
  }

  if (solution_size > 0) {
    MPI_Bcast(solution.data(), solution_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}

void BroadcastDimensions(int &equations_count, int &augmented_columns) {
  MPI_Bcast(&equations_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&augmented_columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void FlattenAndBroadcastMatrix(const std::vector<std::vector<double>> &matrix, int equations_count,
                               int augmented_columns) {
  std::vector<double> flat_matrix(static_cast<size_t>(equations_count * augmented_columns));

  for (int i = 0; i < equations_count; ++i) {
    for (int j = 0; j < augmented_columns; ++j) {
      size_t index = (static_cast<size_t>(i) * static_cast<size_t>(augmented_columns)) + static_cast<size_t>(j);
      flat_matrix[index] = matrix[i][j];
    }
  }

  MPI_Bcast(flat_matrix.data(), equations_count * augmented_columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void ReceiveAndReconstructMatrix(std::vector<std::vector<double>> &matrix, int equations_count, int augmented_columns) {
  std::vector<double> flat_matrix(static_cast<size_t>(equations_count * augmented_columns));
  MPI_Bcast(flat_matrix.data(), equations_count * augmented_columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  matrix.resize(static_cast<size_t>(equations_count));
  for (int i = 0; i < equations_count; ++i) {
    matrix[i].resize(static_cast<size_t>(augmented_columns));
    for (int j = 0; j < augmented_columns; ++j) {
      size_t index = (static_cast<size_t>(i) * static_cast<size_t>(augmented_columns)) + static_cast<size_t>(j);
      matrix[i][j] = flat_matrix[index];
    }
  }
}

}  // namespace

GaussJordanMPI::GaussJordanMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType matrix_copy(in);
  GetInput().swap(matrix_copy);
}

bool GaussJordanMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &input_matrix = GetInput();

    if (input_matrix.empty()) {
      int valid_flag = 1;
      MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return true;
    }

    size_t columns = input_matrix[0].size();
    if (columns == 0) {
      int valid_flag = 0;
      MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return false;
    }

    size_t total_elements = 0;
    for (const auto &row : input_matrix) {
      total_elements += row.size();
      if (row.size() != columns) {
        int valid_flag = 0;
        MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return false;
      }
    }

    bool is_valid = GetOutput().empty() && (total_elements == (columns * input_matrix.size()));
    int valid_flag = is_valid ? 1 : 0;
    MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return is_valid;
  }

  int valid_flag = 0;
  MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return valid_flag != 0;
}

bool GaussJordanMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  GetOutput().clear();

  int equations_count = 0;
  int augmented_columns = 0;

  if (rank == 0) {
    const auto &input_matrix = GetInput();
    if (input_matrix.empty()) {
      equations_count = 0;
      augmented_columns = 0;
    } else {
      equations_count = static_cast<int>(input_matrix.size());
      augmented_columns = static_cast<int>(input_matrix[0].size());
    }
  }

  BroadcastDimensions(equations_count, augmented_columns);

  if (equations_count == 0 || augmented_columns == 0) {
    return true;
  }

  if (rank == 0) {
    FlattenAndBroadcastMatrix(GetInput(), equations_count, augmented_columns);
  } else {
    ReceiveAndReconstructMatrix(GetInput(), equations_count, augmented_columns);
  }

  return true;
}

bool GaussJordanMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (GetInput().empty()) {
    HandleRankZeroOutput(rank, GetOutput(), std::vector<double>());
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  std::vector<std::vector<double>> augmented_matrix = GetInput();
  int equations_count = static_cast<int>(augmented_matrix.size());
  int augmented_columns = (equations_count > 0) ? static_cast<int>(augmented_matrix[0].size()) : 0;

  if (ShouldReturnEarly(rank, augmented_matrix, equations_count, augmented_columns, GetOutput())) {
    return true;
  }

  TransformToReducedRowEchelonFormMPI(augmented_matrix, equations_count, augmented_columns);

  bool global_inconsistent = false;
  int global_rank = 0;
  std::vector<double> local_solution;

  ProcessReducedMatrix(augmented_matrix, equations_count, augmented_columns, global_inconsistent, global_rank,
                       local_solution, rank);

  if (rank == 0) {
    if (global_inconsistent || !HasValidRank(global_rank, augmented_columns, equations_count)) {
      GetOutput() = std::vector<double>();
    } else {
      GetOutput() = local_solution;
    }
  }

  BroadcastSolution(GetOutput(), rank);
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool GaussJordanMPI::PostProcessingImpl() {
  return true;
}
}  // namespace gauss_jordan
