#pragma once

#include <vector>

#include "kosolapov_v_gauss_method_tape_hor_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kosolapov_v_gauss_method_tape_hor_scheme {

class KosolapovVGaussMethodTapeHorSchemeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KosolapovVGaussMethodTapeHorSchemeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void ForwardElimination(std::vector<std::vector<double>> &a, std::vector<double> &b,
                                 std::vector<int> &col_order, int n);
  static std::vector<double> BackwardSubstitution(std::vector<std::vector<double>> &a, std::vector<double> &b,
                                                  std::vector<int> &col_order, int n);
  static void SelectPivot(int i, int n, const std::vector<std::vector<double>> &a, double &max_elem, int &leading_col);
  static void SwapRows(int leading_col, int n, int i, std::vector<std::vector<double>> &a, std::vector<int> &col_order);
  static void RowSub(int i, int n, std::vector<std::vector<double>> &a, std::vector<double> &b);
};

}  // namespace kosolapov_v_gauss_method_tape_hor_scheme
