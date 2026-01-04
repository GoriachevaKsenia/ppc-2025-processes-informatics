#pragma once

#include "gonozov_l_simple_iteration_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace gonozov_l_simple_iteration_method {

class GonozovLSimpleIterationMethodSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit GonozovLSimpleIterationMethodSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int number_unknowns_;
};

}  // namespace gonozov_l_simple_iteration_method
