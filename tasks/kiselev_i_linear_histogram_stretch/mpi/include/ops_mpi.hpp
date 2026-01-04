#pragma once

#include "kiselev_i_linear_histogram_stretch/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kiselev_i_linear_histogram_stretch {

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

}  // namespace kiselev_i_linear_histogram_stretch
