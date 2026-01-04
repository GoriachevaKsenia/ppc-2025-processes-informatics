#pragma once

#include "kiselev_i_gauss_method_horizontal_tape_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kiselev_i_gauss_method_horizontal_tape_scheme {

class KiselevITestTaskMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KiselevITestTaskMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kiselev_i_gauss_method_horizontal_tape_scheme
