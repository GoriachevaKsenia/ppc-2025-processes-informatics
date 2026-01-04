#pragma once

#include <vector>

#include "kosolapov_v_gauss_method_tape_hor_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kosolapov_v_gauss_method_tape_hor_scheme {

class KosolapovVGaussMethodTapeHorSchemeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KosolapovVGaussMethodTapeHorSchemeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  void DistributeDataFromRoot(std::vector<std::vector<double>> &local_matrix, std::vector<double> &local_rsd,
                              std::vector<int> &local_row_indices, int start, int local_rows, int columns,
                              int processes_count, int rows_per_proc, int remainder);
  bool ForwardElimination(int rank, int rows, int start_row, int end_row, int local_rows,
                          std::vector<std::vector<double>> &local_matrix, std::vector<int> &col_order,
                          std::vector<double> &local_rsd, std::vector<int> &local_row_indices);
  static std::vector<double> BackwardSubstitution(int rows, int start_row, int end_row, int rank, int local_rows,
                                                  std::vector<std::vector<double>> &local_matrix,
                                                  std::vector<double> &local_rsd);
  static void SelectPivot(int rank, int owner_process, int step, int start_row, int local_rows, int rows,
                          const std::vector<std::vector<double>> &local_matrix, double &pivot_value, int &pivot_col,
                          int &pivot_row_local_idx);
  static void RowSub(int local_rows, int step, int rows, std::vector<std::vector<double>> &local_matrix,
                     std::vector<double> &local_rsd, const std::vector<int> &local_row_indices,
                     const std::vector<double> &pivot_row, double pivot_rhs);
  static void SwapColumns(int pivot_col, int step, int local_rows, std::vector<std::vector<double>> &local_matrix,
                          std::vector<int> &col_order);
  static void NormalizePivot(int rank, int owner_process, int pivot_row_local_idx, int step, int rows,
                             double pivot_value, std::vector<std::vector<double>> &local_matrix,
                             std::vector<double> &local_rsd);
  static void PriparePivot(int rank, int owner_process, int pivot_row_local_idx, int step, int rows,
                           std::vector<double> &pivot_row, double &pivot_rhs,
                           const std::vector<std::vector<double>> &local_matrix, const std::vector<double> &local_rsd);
};

}  // namespace kosolapov_v_gauss_method_tape_hor_scheme
