#pragma once

#include "pylaeva_s_simple_iteration_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace pylaeva_s_simple_iteration_method {

class PylaevaSSimpleIterationMethodSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PylaevaSSimpleIterationMethodSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace pylaeva_s_simple_iteration_method
