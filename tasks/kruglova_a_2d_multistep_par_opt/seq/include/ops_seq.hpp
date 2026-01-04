#pragma once

#include "kruglova_a_2d_multistep_par_opt/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kruglova_a_2d_multistep_par_opt {

class KruglovaA2DMuitSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KruglovaA2DMuitSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kruglova_a_2d_multistep_par_opt
