#include "lobanov_d_multiply_matrix_ccs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "lobanov_d_multiply_matrix_ccs/common/include/common.hpp"

namespace lobanov_d_multiply_matrix_ccs {

namespace {
constexpr double kEpsilonThreshold = 1e-10;

std::vector<int> CountElementsPerRow(const CompressedColumnMatrix &source_matrix, int column_count) {
  std::vector<int> row_element_count(column_count, 0);

  for (int element_index = 0; element_index < source_matrix.non_zero_count; element_index++) {
    int row_idx = source_matrix.row_index_data[element_index];
    if (row_idx >= 0 && row_idx < column_count) {
      row_element_count[row_idx]++;
    }
  }

  return row_element_count;
}

void BuildColumnPointers(CompressedColumnMatrix &transposed_result, const std::vector<int> &row_element_count) {
  transposed_result.column_pointer_data.resize(transposed_result.column_count + 1);
  transposed_result.column_pointer_data[0] = 0;

  for (int row_index = 0; row_index < transposed_result.column_count; row_index++) {
    transposed_result.column_pointer_data[row_index + 1] =
        transposed_result.column_pointer_data[row_index] + row_element_count[row_index];
  }
}

void FillTransposedMatrixData(const CompressedColumnMatrix &source_matrix, CompressedColumnMatrix &transposed_result) {
  transposed_result.value_data.resize(source_matrix.non_zero_count);
  transposed_result.row_index_data.resize(source_matrix.non_zero_count);

  std::vector<int> current_position_array(transposed_result.column_count, 0);

  for (int column_counter = 0; column_counter < source_matrix.column_count; column_counter++) {
    int column_start = source_matrix.column_pointer_data[column_counter];
    int column_end = source_matrix.column_pointer_data[column_counter + 1];

    for (int element_counter = column_start; element_counter < column_end; element_counter++) {
      int row_position = source_matrix.row_index_data[element_counter];
      double element_value = source_matrix.value_data[element_counter];

      if (row_position >= 0 && row_position < transposed_result.column_count) {
        int target_slot = transposed_result.column_pointer_data[row_position] + current_position_array[row_position];

        if (target_slot >= 0 && target_slot < source_matrix.non_zero_count) {
          transposed_result.value_data[target_slot] = element_value;
          transposed_result.row_index_data[target_slot] = column_counter;
        }

        current_position_array[row_position]++;
      }
    }
  }
}

bool ValidateInputParameters(int column_index, const std::vector<int> &local_column_pointers,
                             const std::vector<double> &local_values, const std::vector<int> &local_row_indices) {
  if (column_index < 0 || column_index >= static_cast<int>(local_column_pointers.size()) - 1) {
    return false;
  }

  if (local_values.size() != local_row_indices.size()) {
    return false;
  }

  return true;
}

bool ValidateTransposedMatrix(const CompressedColumnMatrix &transposed_matrix_a) {
  if (transposed_matrix_a.column_count <= 0 || transposed_matrix_a.row_count <= 0) {
    return false;
  }

  if (transposed_matrix_a.column_pointer_data.size() !=
      static_cast<std::size_t>(transposed_matrix_a.column_count) + 1) {
    return false;
  }

  return true;
}

bool ValidateColumnRange(int column_start_position, int column_end_position, const std::vector<double> &local_values) {
  return column_start_position >= 0 && static_cast<std::size_t>(column_end_position) <= local_values.size() &&
         column_start_position <= column_end_position;
}

bool ValidateTemporaryArrays(const std::vector<double> &temporary_row_values, const std::vector<int> &row_marker_array,
                             int column_count) {
  return temporary_row_values.size() == static_cast<std::size_t>(column_count) &&
         row_marker_array.size() == static_cast<std::size_t>(column_count);
}

bool ValidateTransposedRange(int transposed_start, int transposed_end, const std::vector<double> &transposed_values) {
  return transposed_start >= 0 && static_cast<std::size_t>(transposed_end) <= transposed_values.size() &&
         transposed_start <= transposed_end;
}

void ProcessTransposedElement(int matrix_a_row, int column_index, double matrix_a_value, double matrix_b_value,
                              std::vector<double> &temporary_row_values, std::vector<int> &row_marker_array) {
  if (row_marker_array[matrix_a_row] != column_index) {
    row_marker_array[matrix_a_row] = column_index;
    temporary_row_values[matrix_a_row] = matrix_a_value * matrix_b_value;
  } else {
    temporary_row_values[matrix_a_row] += matrix_a_value * matrix_b_value;
  }
}

void ProcessMatrixBElement(int matrix_b_row, double matrix_b_value, const CompressedColumnMatrix &transposed_matrix_a,
                           int column_index, std::vector<double> &temporary_row_values,
                           std::vector<int> &row_marker_array) {
  int transposed_start = transposed_matrix_a.column_pointer_data[matrix_b_row];
  int transposed_end = transposed_matrix_a.column_pointer_data[matrix_b_row + 1];

  if (!ValidateTransposedRange(transposed_start, transposed_end, transposed_matrix_a.value_data)) {
    return;
  }

  for (int transposed_index = transposed_start; transposed_index < transposed_end; transposed_index++) {
    int matrix_a_row = transposed_matrix_a.row_index_data[transposed_index];
    double matrix_a_value = transposed_matrix_a.value_data[transposed_index];

    ProcessTransposedElement(matrix_a_row, column_index, matrix_a_value, matrix_b_value, temporary_row_values,
                             row_marker_array);
  }
}

void ProcessLocalColumnElements(int column_start_position, int column_end_position,
                                const std::vector<double> &local_values, const std::vector<int> &local_row_indices,
                                const CompressedColumnMatrix &transposed_matrix_a, int column_index,
                                std::vector<double> &temporary_row_values, std::vector<int> &row_marker_array) {
  for (int element_index = column_start_position; element_index < column_end_position; element_index++) {
    int matrix_b_row = local_row_indices[element_index];
    double matrix_b_value = local_values[element_index];

    ProcessMatrixBElement(matrix_b_row, matrix_b_value, transposed_matrix_a, column_index, temporary_row_values,
                          row_marker_array);
  }
}

void CollectNonZeroResults(const std::vector<double> &temporary_row_values, const std::vector<int> &row_marker_array,
                           int column_index, int column_count, std::vector<double> &result_values,
                           std::vector<int> &result_row_indices) {
  for (int row_index = 0; row_index < column_count; row_index++) {
    if (row_marker_array[row_index] == column_index && std::abs(temporary_row_values[row_index]) > kEpsilonThreshold) {
      result_values.push_back(temporary_row_values[row_index]);
      result_row_indices.push_back(row_index);
    }
  }
}

bool ValidateInputRange(int start_column, int end_column, int column_count) {
  if (start_column < 0 || end_column < 0 || start_column >= end_column) {
    return false;
  }

  if (start_column >= column_count) {
    return false;
  }

  return true;
}

bool ValidateMatrixB(const CompressedColumnMatrix &matrix_b) {
  if (matrix_b.column_count <= 0 || matrix_b.row_count <= 0) {
    return false;
  }

  if (matrix_b.column_pointer_data.size() != static_cast<std::size_t>(matrix_b.column_count) + 1) {
    return false;
  }

  return true;
}

int ExtractSingleColumnData(int col_index, const CompressedColumnMatrix &matrix_b, std::vector<double> &local_values,
                            std::vector<int> &local_row_indices) {
  int col_start = matrix_b.column_pointer_data[col_index];
  int col_end = matrix_b.column_pointer_data[col_index + 1];

  for (int idx = col_start; idx < col_end; ++idx) {
    int row_idx = matrix_b.row_index_data[idx];

    local_values.push_back(matrix_b.value_data[idx]);
    local_row_indices.push_back(row_idx);
  }

  return static_cast<int>(local_values.size());
}

bool ValidateProcessCount(int total_processes) {
  return total_processes > 0;
}

void InitializeResultMatrix(const CompressedColumnMatrix &matrix_a, const CompressedColumnMatrix &matrix_b,
                            CompressedColumnMatrix &result_matrix) {
  result_matrix.row_count = matrix_a.row_count;
  result_matrix.column_count = matrix_b.column_count;
  result_matrix.non_zero_count = 0;
  result_matrix.value_data.clear();
  result_matrix.row_index_data.clear();
  result_matrix.column_pointer_data.clear();
  result_matrix.column_pointer_data.push_back(0);
}

bool ReceiveProcessDataMPI(int process_id, std::vector<double> &values, std::vector<int> &row_indices,
                           std::vector<int> &col_pointers) {
  int nnz = 0;
  int cols = 0;
  MPI_Status status;

  if (MPI_Recv(&nnz, 1, MPI_INT, process_id, 0, MPI_COMM_WORLD, &status) != MPI_SUCCESS) {
    return false;
  }
  if (MPI_Recv(&cols, 1, MPI_INT, process_id, 1, MPI_COMM_WORLD, &status) != MPI_SUCCESS) {
    return false;
  }

  const int max_size = 100000000;
  if (nnz < 0 || cols < 0 || nnz > max_size || cols > max_size) {
    return false;
  }

  values.resize(nnz);
  row_indices.resize(nnz);
  col_pointers.resize(cols + 1);

  if (nnz > 0) {
    if (MPI_Recv(values.data(), nnz, MPI_DOUBLE, process_id, 2, MPI_COMM_WORLD, &status) != MPI_SUCCESS) {
      return false;
    }
    if (MPI_Recv(row_indices.data(), nnz, MPI_INT, process_id, 3, MPI_COMM_WORLD, &status) != MPI_SUCCESS) {
      return false;
    }
  }

  if (MPI_Recv(col_pointers.data(), cols + 1, MPI_INT, process_id, 4, MPI_COMM_WORLD, &status) != MPI_SUCCESS) {
    return false;
  }

  return true;
}

void ProcessLocalWorkerData(int pid, const std::vector<std::vector<double>> &value_collections,
                            const std::vector<std::vector<int>> &row_index_collections,
                            const std::vector<std::vector<int>> &column_pointer_collections, int &value_offset,
                            CompressedColumnMatrix &result_matrix) {
  if (value_collections[pid].size() != row_index_collections[pid].size()) {
    return;
  }

  if (column_pointer_collections[pid].empty()) {
    return;
  }

  if (!value_collections[pid].empty()) {
    result_matrix.value_data.insert(result_matrix.value_data.end(), value_collections[pid].begin(),
                                    value_collections[pid].end());

    result_matrix.row_index_data.insert(result_matrix.row_index_data.end(), row_index_collections[pid].begin(),
                                        row_index_collections[pid].end());
  }

  for (size_t i = 1; i < column_pointer_collections[pid].size(); ++i) {
    int adjusted_value = std::max(column_pointer_collections[pid][i], 0);
    result_matrix.column_pointer_data.push_back(adjusted_value + value_offset);
  }

  value_offset += static_cast<int>(value_collections[pid].size());
}

void ReceiveAllWorkerData(int total_processes, std::vector<std::vector<double>> &value_collections,
                          std::vector<std::vector<int>> &row_index_collections,
                          std::vector<std::vector<int>> &column_pointer_collections,
                          std::vector<double> &local_result_values, std::vector<int> &local_result_row_indices,
                          std::vector<int> &local_result_column_pointers) {
  value_collections[0] = std::move(local_result_values);
  row_index_collections[0] = std::move(local_result_row_indices);
  column_pointer_collections[0] = std::move(local_result_column_pointers);

  for (int pid = 1; pid < total_processes; ++pid) {
    if (!ReceiveProcessDataMPI(pid, value_collections[pid], row_index_collections[pid],
                               column_pointer_collections[pid])) {
      value_collections[pid].clear();
      row_index_collections[pid].clear();
      column_pointer_collections[pid].clear();
    }
  }
}

void FinalizeResultMatrix(CompressedColumnMatrix &result_matrix) {
  result_matrix.non_zero_count = static_cast<int>(result_matrix.value_data.size());

  if (result_matrix.column_pointer_data.size() != static_cast<std::size_t>(result_matrix.column_count) + 1) {
    result_matrix.column_pointer_data.resize(result_matrix.column_count + 1);
    if (!result_matrix.column_pointer_data.empty()) {
      result_matrix.column_pointer_data[0] = 0;
    }
  }
}

}  // namespace

LobanovDMultiplyMatrixMPI::LobanovDMultiplyMatrixMPI(const InType &input_matrices) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input_matrices;
  CompressedColumnMatrix empty_matrix;
  empty_matrix.row_count = 0;
  empty_matrix.column_count = 0;
  empty_matrix.non_zero_count = 0;
  GetOutput() = empty_matrix;
}

bool LobanovDMultiplyMatrixMPI::ValidationImpl() {
  const auto &[matrix_a, matrix_b] = GetInput();
  return (matrix_a.column_count == matrix_b.row_count && matrix_a.row_count > 0 && matrix_a.column_count > 0 &&
          matrix_b.row_count > 0 && matrix_b.column_count > 0);
}

bool LobanovDMultiplyMatrixMPI::PreProcessingImpl() {
  return true;
}

void LobanovDMultiplyMatrixMPI::ComputeTransposedMatrixMPI(const CompressedColumnMatrix &source_matrix,
                                                           CompressedColumnMatrix &transposed_result) {
  transposed_result.ZeroInitialize();

  transposed_result.row_count = source_matrix.column_count;
  transposed_result.column_count = source_matrix.row_count;
  transposed_result.non_zero_count = source_matrix.non_zero_count;

  std::vector<int> row_element_count = CountElementsPerRow(source_matrix, transposed_result.column_count);

  BuildColumnPointers(transposed_result, row_element_count);

  FillTransposedMatrixData(source_matrix, transposed_result);
}

std::pair<int, int> LobanovDMultiplyMatrixMPI::DetermineColumnDistribution(int total_columns, int process_rank,
                                                                           int process_count) {
  int base_columns_per_process = total_columns / process_count;
  int remaining_columns = total_columns % process_count;

  int start_column = (process_rank * base_columns_per_process) + std::min(process_rank, remaining_columns);
  int end_column = start_column + base_columns_per_process + (process_rank < remaining_columns ? 1 : 0);

  start_column = std::max(0, std::min(start_column, total_columns));
  end_column = std::max(0, std::min(end_column, total_columns));

  start_column = std::min(start_column, end_column);

  return {start_column, end_column};
}

void LobanovDMultiplyMatrixMPI::ProcessLocalColumnMPI(
    const CompressedColumnMatrix &transposed_matrix_a, const std::vector<double> &local_values,
    const std::vector<int> &local_row_indices, const std::vector<int> &local_column_pointers, int column_index,
    std::vector<double> &temporary_row_values, std::vector<int> &row_marker_array, std::vector<double> &result_values,
    std::vector<int> &result_row_indices) {
  if (!ValidateInputParameters(column_index, local_column_pointers, local_values, local_row_indices)) {
    return;
  }

  if (!ValidateTransposedMatrix(transposed_matrix_a)) {
    return;
  }

  int column_start_position = local_column_pointers[column_index];
  int column_end_position = local_column_pointers[column_index + 1];

  if (!ValidateColumnRange(column_start_position, column_end_position, local_values)) {
    return;
  }

  if (!ValidateTemporaryArrays(temporary_row_values, row_marker_array, transposed_matrix_a.column_count)) {
    return;
  }

  ProcessLocalColumnElements(column_start_position, column_end_position, local_values, local_row_indices,
                             transposed_matrix_a, column_index, temporary_row_values, row_marker_array);

  CollectNonZeroResults(temporary_row_values, row_marker_array, column_index, transposed_matrix_a.column_count,
                        result_values, result_row_indices);
}

void LobanovDMultiplyMatrixMPI::ExtractLocalColumnData(const CompressedColumnMatrix &matrix_b, int start_column,
                                                       int end_column, std::vector<double> &local_values,
                                                       std::vector<int> &local_row_indices,
                                                       std::vector<int> &local_column_pointers) {
  local_values.clear();
  local_row_indices.clear();
  local_column_pointers.clear();
  local_column_pointers.push_back(0);

  if (!ValidateInputRange(start_column, end_column, matrix_b.column_count)) {
    return;
  }

  if (!ValidateMatrixB(matrix_b)) {
    return;
  }

  end_column = std::min(end_column, matrix_b.column_count);

  for (int current_col = start_column; current_col < end_column; ++current_col) {
    int new_size = ExtractSingleColumnData(current_col, matrix_b, local_values, local_row_indices);
    local_column_pointers.push_back(new_size);
  }
}

void LobanovDMultiplyMatrixMPI::MultiplyLocalMatricesMPI(const CompressedColumnMatrix &transposed_matrix_a,
                                                         const std::vector<double> &local_values,
                                                         const std::vector<int> &local_row_indices,
                                                         const std::vector<int> &local_column_pointers,
                                                         int local_column_count, std::vector<double> &result_values,
                                                         std::vector<int> &result_row_indices,
                                                         std::vector<int> &result_column_pointers) {
  result_values.clear();
  result_row_indices.clear();
  result_column_pointers.clear();
  result_column_pointers.push_back(0);

  if (transposed_matrix_a.column_count <= 0) {
    return;
  }

  if (local_values.size() != local_row_indices.size()) {
    return;
  }

  if (local_column_pointers.size() != static_cast<std::size_t>(local_column_count) + 1) {
    return;
  }

  if (transposed_matrix_a.column_count > 1000000) {  // разумный предел
    return;
  }

  int matrix_rows = transposed_matrix_a.column_count;

  std::vector<double> temp_row_buffer(matrix_rows, 0.0);
  std::vector<int> row_tracker(matrix_rows, -1);

  int column_counter = 0;
  while (column_counter < local_column_count) {
    ProcessLocalColumnMPI(transposed_matrix_a, local_values, local_row_indices, local_column_pointers, column_counter,
                          temp_row_buffer, row_tracker, result_values, result_row_indices);

    result_column_pointers.push_back(static_cast<int>(result_values.size()));
    column_counter++;
  }
}

bool LobanovDMultiplyMatrixMPI::ProcessMasterRank(const CompressedColumnMatrix &matrix_a,
                                                  const CompressedColumnMatrix &matrix_b,
                                                  std::vector<double> &local_result_values,
                                                  std::vector<int> &local_result_row_indices,
                                                  std::vector<int> &local_result_column_pointers, int total_processes) {
  if (!ValidateProcessCount(total_processes)) {
    return false;
  }

  CompressedColumnMatrix result_matrix;
  InitializeResultMatrix(matrix_a, matrix_b, result_matrix);

  std::vector<std::vector<double>> value_collections(total_processes);
  std::vector<std::vector<int>> row_index_collections(total_processes);
  std::vector<std::vector<int>> column_pointer_collections(total_processes);

  ReceiveAllWorkerData(total_processes, value_collections, row_index_collections, column_pointer_collections,
                       local_result_values, local_result_row_indices, local_result_column_pointers);

  int value_offset = 0;
  for (int pid = 0; pid < total_processes; ++pid) {
    ProcessLocalWorkerData(pid, value_collections, row_index_collections, column_pointer_collections, value_offset,
                           result_matrix);
  }

  FinalizeResultMatrix(result_matrix);

  GetOutput() = result_matrix;

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool LobanovDMultiplyMatrixMPI::ProcessWorkerRank(const std::vector<double> &local_result_values,
                                                  const std::vector<int> &local_result_row_indices,
                                                  const std::vector<int> &local_result_column_pointers,
                                                  int local_column_count) {
  int local_non_zero_count = static_cast<int>(local_result_values.size());
  int local_column_total = local_column_count;

  MPI_Send(&local_non_zero_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  MPI_Send(&local_column_total, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
  MPI_Send(local_result_values.data(), local_non_zero_count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
  MPI_Send(local_result_row_indices.data(), local_non_zero_count, MPI_INT, 0, 3, MPI_COMM_WORLD);
  MPI_Send(local_result_column_pointers.data(), local_column_count + 1, MPI_INT, 0, 4, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool LobanovDMultiplyMatrixMPI::RunImpl() {
  int process_rank = 0;
  int total_processes = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &total_processes);
  const auto &[matrix_a, matrix_b] = GetInput();

  CompressedColumnMatrix transposed_matrix_a;
  transposed_matrix_a.row_count = 0;
  transposed_matrix_a.column_count = 0;
  transposed_matrix_a.non_zero_count = 0;
  transposed_matrix_a.value_data.clear();
  transposed_matrix_a.row_index_data.clear();
  transposed_matrix_a.column_pointer_data.clear();

  if (process_rank == 0) {
    ComputeTransposedMatrixMPI(matrix_a, transposed_matrix_a);

    std::array<int, 3> transposed_dims = {transposed_matrix_a.row_count, transposed_matrix_a.column_count,
                                          transposed_matrix_a.non_zero_count};

    MPI_Bcast(transposed_dims.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);

    if (transposed_dims[2] > 0) {
      MPI_Bcast(transposed_matrix_a.value_data.data(), transposed_dims[2], MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(transposed_matrix_a.row_index_data.data(), transposed_dims[2], MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Bcast(transposed_matrix_a.column_pointer_data.data(), transposed_dims[1] + 1, MPI_INT, 0, MPI_COMM_WORLD);

    transposed_matrix_a.row_count = transposed_dims[0];
    transposed_matrix_a.column_count = transposed_dims[1];
    transposed_matrix_a.non_zero_count = transposed_dims[2];
  } else {
    std::array<int, 3> transposed_dims{};
    MPI_Bcast(transposed_dims.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);

    transposed_matrix_a.row_count = transposed_dims[0];
    transposed_matrix_a.column_count = transposed_dims[1];
    transposed_matrix_a.non_zero_count = transposed_dims[2];

    if (transposed_matrix_a.non_zero_count > 0) {
      transposed_matrix_a.value_data.resize(static_cast<std::size_t>(transposed_matrix_a.non_zero_count), 0.0);
      transposed_matrix_a.row_index_data.resize(static_cast<std::size_t>(transposed_matrix_a.non_zero_count), 0);

      MPI_Bcast(transposed_matrix_a.value_data.data(), transposed_matrix_a.non_zero_count, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
      MPI_Bcast(transposed_matrix_a.row_index_data.data(), transposed_matrix_a.non_zero_count, MPI_INT, 0,
                MPI_COMM_WORLD);
    }

    transposed_matrix_a.column_pointer_data.resize(static_cast<std::size_t>(transposed_matrix_a.column_count) + 1U, 0);
    MPI_Bcast(transposed_matrix_a.column_pointer_data.data(), transposed_matrix_a.column_count + 1, MPI_INT, 0,
              MPI_COMM_WORLD);
  }

  std::array<int, 2> columns_per_process = {0, 0};

  if (process_rank == 0) {
    auto [start, end] = DetermineColumnDistribution(matrix_b.column_count, process_rank, total_processes);
    columns_per_process[0] = start;
    columns_per_process[1] = end;

    for (int pid = 1; pid < total_processes; ++pid) {
      auto [pid_start, pid_end] = DetermineColumnDistribution(matrix_b.column_count, pid, total_processes);

      std::array<int, 2> pid_columns = {pid_start, pid_end};
      MPI_Send(pid_columns.data(), 2, MPI_INT, pid, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(columns_per_process.data(), 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  int start_column = columns_per_process[0];
  int end_column = columns_per_process[1];
  int local_column_count = end_column - start_column;

  std::vector<double> local_values;
  std::vector<int> local_row_indices;
  std::vector<int> local_column_pointers;

  ExtractLocalColumnData(matrix_b, start_column, end_column, local_values, local_row_indices, local_column_pointers);

  std::vector<double> result_values;
  std::vector<int> result_row_indices;
  std::vector<int> result_column_pointers;

  MultiplyLocalMatricesMPI(transposed_matrix_a, local_values, local_row_indices, local_column_pointers,
                           local_column_count, result_values, result_row_indices, result_column_pointers);

  if (process_rank == 0) {
    return ProcessMasterRank(matrix_a, matrix_b, result_values, result_row_indices, result_column_pointers,
                             total_processes);
  }
  return ProcessWorkerRank(result_values, result_row_indices, result_column_pointers, local_column_count);
}

bool LobanovDMultiplyMatrixMPI::PostProcessingImpl() {
  int current_process_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &current_process_rank);

  const CompressedColumnMatrix &computed_result = GetOutput();

  if (current_process_rank == 0) {
    bool dimensions_valid = computed_result.row_count > 0 && computed_result.column_count > 0;
    bool structure_valid =
        computed_result.column_pointer_data.size() == static_cast<std::size_t>(computed_result.column_count) + 1U;
    return dimensions_valid && structure_valid;
  }

  bool is_empty = computed_result.row_count == 0 && computed_result.column_count == 0;
  return is_empty;
}

}  // namespace lobanov_d_multiply_matrix_ccs
