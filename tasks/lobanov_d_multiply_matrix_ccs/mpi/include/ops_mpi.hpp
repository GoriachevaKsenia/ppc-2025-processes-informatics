#pragma once

#include <utility>
#include <vector>

#include "lobanov_d_multiply_matrix_ccs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lobanov_d_multiply_matrix_ccs {

class LobanovDMultiplyMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LobanovDMultiplyMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void ComputeTransposedMatrixMPI(const CompressedColumnMatrix &source_matrix,
                                         CompressedColumnMatrix &transposed_result);
  static std::pair<int, int> DetermineColumnDistribution(int total_columns, int process_rank, int process_count);
  static void ExtractLocalColumnData(const CompressedColumnMatrix &matrix_b, int start_column, int end_column,
                                     std::vector<double> &local_values, std::vector<int> &local_row_indices,
                                     std::vector<int> &local_column_pointers);
  static void MultiplyLocalMatricesMPI(const CompressedColumnMatrix &transposed_matrix_a,
                                       const std::vector<double> &local_values,
                                       const std::vector<int> &local_row_indices,
                                       const std::vector<int> &local_column_pointers, int local_column_count,
                                       std::vector<double> &result_values, std::vector<int> &result_row_indices,
                                       std::vector<int> &result_column_pointers);
  bool ProcessMasterRank(const CompressedColumnMatrix &matrix_a, const CompressedColumnMatrix &matrix_b,
                         std::vector<double> &local_result_values, std::vector<int> &local_result_row_indices,
                         std::vector<int> &local_result_column_pointers, int total_processes);
  static bool ProcessWorkerRank(const std::vector<double> &local_result_values,
                                const std::vector<int> &local_result_row_indices,
                                const std::vector<int> &local_result_column_pointers, int local_column_count);
  static void ProcessLocalColumnMPI(const CompressedColumnMatrix &transposed_matrix_a,
                                    const std::vector<double> &local_values, const std::vector<int> &local_row_indices,
                                    const std::vector<int> &local_column_pointers, int column_index,
                                    std::vector<double> &temporary_row_values, std::vector<int> &row_marker_array,
                                    std::vector<double> &result_values, std::vector<int> &result_row_indices);
};

}  // namespace lobanov_d_multiply_matrix_ccs
