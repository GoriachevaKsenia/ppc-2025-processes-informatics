#pragma once

#include <vector>

#include "khruev_a_gauss_jordan/common/include/common.hpp"
#include "task/include/task.hpp"

namespace khruev_a_gauss_jordan {

class KhruevAGaussJordanSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit KhruevAGaussJordanSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void ToReducedForm(std::vector<std::vector<double>> &a);
  static bool DetectInconsistency(const std::vector<std::vector<double>> &a);
  static int ComputeRank(const std::vector<std::vector<double>> &a);
  static std::vector<double> RecoverSolution(const std::vector<std::vector<double>> &a);
};

}  // namespace khruev_a_gauss_jordan
