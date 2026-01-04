#pragma once

#include <vector>

#include "../../common/include/common.hpp"
#include "task/include/task.hpp"

namespace shekhirev_v_cg_method_seq {

class ConjugateGradientSeq : public shekhirev_v_cg_method::BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ConjugateGradientSeq(const shekhirev_v_cg_method::InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::vector<double> MultiplyMatrixVector(const std::vector<double> &matrix, const std::vector<double> &vec,
                                                  int n);
  static double DotProduct(const std::vector<double> &a, const std::vector<double> &b);
};

}  // namespace shekhirev_v_cg_method_seq
