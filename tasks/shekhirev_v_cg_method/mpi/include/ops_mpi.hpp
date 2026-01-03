#pragma once

#include <vector>

#include "../../common/include/common.hpp"
#include "task/include/task.hpp"

namespace shekhirev_v_cg_method_mpi {

class ConjugateGradientMPI : public shekhirev_v_cg_method::BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ConjugateGradientMPI(const shekhirev_v_cg_method::InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::vector<double> LocalMultiply(const std::vector<double> &local_a, const std::vector<double> &global_p,
                                           int local_n, int n);
  static double LocalDotProduct(const std::vector<double> &a, const std::vector<double> &b);
};

}  // namespace shekhirev_v_cg_method_mpi
