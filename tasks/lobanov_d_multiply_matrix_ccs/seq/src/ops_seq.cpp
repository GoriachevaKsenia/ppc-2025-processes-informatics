#include "lobanov_d_multiply_matrix_ccs/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <vector>

#include "lobanov_d_multiply_matrix_ccs/common/include/common.hpp"

namespace lobanov_d_multiply_matrix_ccs {

constexpr double kEpsilonThreshold = 1e-12;

LobanovDMultiplyMatrixSEQ::LobanovDMultiplyMatrixSEQ(const InType &input_matrices) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input_matrices;
  CompressedColumnMatrix empty_matrix;
  empty_matrix.row_count = 0;
  empty_matrix.column_count = 0;
  empty_matrix.non_zero_count = 0;
  GetOutput() = empty_matrix;
}

bool LobanovDMultiplyMatrixSEQ::ValidationImpl() {
  const auto &[matrix_a, matrix_b] = GetInput();
  return (matrix_a.row_count > 0 && matrix_a.column_count > 0 && matrix_b.row_count > 0 && matrix_b.column_count > 0 &&
          matrix_a.column_count == matrix_b.row_count);
}

bool LobanovDMultiplyMatrixSEQ::PreProcessingImpl() {
  return true;
}

void LobanovDMultiplyMatrixSEQ::ComputeTransposedMatrix(const CompressedColumnMatrix &source_matrix,
                                                        CompressedColumnMatrix &transposed_result) {
  transposed_result.row_count = source_matrix.column_count;
  transposed_result.column_count = source_matrix.row_count;
  transposed_result.non_zero_count = source_matrix.non_zero_count;

  if (source_matrix.non_zero_count == 0) {
    transposed_result.value_data.clear();
    transposed_result.row_index_data.clear();
    transposed_result.column_pointer_data = std::vector<int>(transposed_result.column_count + 1);
    return;
  }

  struct Element {
    int new_col;
    int new_row;
    double value;
  };

  std::vector<Element> elements;
  elements.reserve(source_matrix.non_zero_count);

  for (int old_col = 0; old_col < source_matrix.column_count; old_col++) {
    for (int pos = source_matrix.column_pointer_data[old_col]; pos < source_matrix.column_pointer_data[old_col + 1];
         pos++) {
      int old_row = source_matrix.row_index_data[pos];
      elements.push_back({old_row, old_col, source_matrix.value_data[pos]});
    }
  }

  std::ranges::sort(elements.begin(), elements.end(), [](const Element &a, const Element &b) {
    return (a.new_col < b.new_col) || (a.new_col == b.new_col && a.new_row < b.new_row);
  });

  transposed_result.value_data.resize(source_matrix.non_zero_count);
  transposed_result.row_index_data.resize(source_matrix.non_zero_count);
  transposed_result.column_pointer_data.resize(transposed_result.column_count + 1);

  transposed_result.column_pointer_data[0] = 0;
  int current_col = 0;
  int idx = 0;

  for (const auto &elem : elements) {
    transposed_result.value_data[idx] = elem.value;
    transposed_result.row_index_data[idx] = elem.new_row;
    idx++;

    while (current_col < elem.new_col) {
      transposed_result.column_pointer_data[current_col + 1] = idx;
      current_col++;
    }
  }

  while (current_col < transposed_result.column_count) {
    transposed_result.column_pointer_data[current_col + 1] = idx;
    current_col++;
  }
}

namespace {

void ProcessMatrixColumn(const CompressedColumnMatrix &transposed_a, const CompressedColumnMatrix &matrix_b,
                         int target_column, std::vector<double> &temporary_values, std::vector<int> &column_markers,
                         std::vector<double> &result_values, std::vector<int> &result_row_indices) {
  for (int b_pointer = matrix_b.column_pointer_data[target_column];
       b_pointer < matrix_b.column_pointer_data[target_column + 1]; b_pointer++) {
    int b_row = matrix_b.row_index_data[b_pointer];
    double b_value = matrix_b.value_data[b_pointer];

    for (int a_pointer = transposed_a.column_pointer_data[b_row];
         a_pointer < transposed_a.column_pointer_data[b_row + 1]; a_pointer++) {
      int a_row = transposed_a.row_index_data[a_pointer];
      double a_value = transposed_a.value_data[a_pointer];

      if (column_markers[a_row] != target_column) {
        column_markers[a_row] = target_column;
        temporary_values[a_row] = a_value * b_value;
      } else {
        temporary_values[a_row] += a_value * b_value;
      }
    }
  }

  for (size_t row_index = 0; row_index < temporary_values.size(); row_index++) {
    if (column_markers[row_index] == target_column && std::abs(temporary_values[row_index]) > kEpsilonThreshold) {
      result_values.push_back(temporary_values[row_index]);
      result_row_indices.push_back(static_cast<int>(row_index));
    }
  }
}

}  // namespace

void LobanovDMultiplyMatrixSEQ::PerformMatrixMultiplication(const CompressedColumnMatrix &first_matrix,
                                                            const CompressedColumnMatrix &second_matrix,
                                                            CompressedColumnMatrix &product_result) {
  CompressedColumnMatrix transposed_first = {};
  transposed_first.row_count = 0;
  transposed_first.column_count = 0;
  transposed_first.non_zero_count = 0;

  ComputeTransposedMatrix(first_matrix, transposed_first);

  product_result.row_count = first_matrix.row_count;
  product_result.column_count = second_matrix.column_count;

  product_result.column_pointer_data = {0};

  int result_rows = product_result.row_count;
  std::vector<double> column_temporary_values(result_rows, 0.0);
  std::vector<int> column_marker_tracker(result_rows, -1);

  int cols_to_process = second_matrix.column_count;
  for (int column_index = 0; column_index < cols_to_process; column_index++) {
    ProcessMatrixColumn(transposed_first, second_matrix, column_index, column_temporary_values, column_marker_tracker,
                        product_result.value_data, product_result.row_index_data);

    product_result.column_pointer_data.push_back(static_cast<int>(product_result.value_data.size()));
  }

  product_result.non_zero_count = static_cast<int>(product_result.value_data.size());
}

bool LobanovDMultiplyMatrixSEQ::RunImpl() {
  const auto &[matrix_a, matrix_b] = GetInput();

  try {
    CompressedColumnMatrix result_matrix;
    result_matrix.row_count = 0;
    result_matrix.column_count = 0;
    result_matrix.non_zero_count = 0;

    PerformMatrixMultiplication(matrix_a, matrix_b, result_matrix);
    GetOutput() = result_matrix;
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool LobanovDMultiplyMatrixSEQ::PostProcessingImpl() {
  const auto &result_matrix = GetOutput();
  return result_matrix.row_count > 0 && result_matrix.column_count > 0 &&
         result_matrix.column_pointer_data.size() == static_cast<size_t>(result_matrix.column_count) + 1;
}

}  // namespace lobanov_d_multiply_matrix_ccs
