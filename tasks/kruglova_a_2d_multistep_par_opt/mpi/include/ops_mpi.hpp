#pragma once

#include <functional>

#include "kruglova_a_2d_multistep_par_opt/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kruglova_a_2d_multistep_par_opt {

class KruglovaA2DMuitMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KruglovaA2DMuitMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  double Solve1DSequential(std::function<double(double)> &func, double a, double b, double eps, int max_iters);
};

}  // namespace kruglova_a_2d_multistep_par_opt
