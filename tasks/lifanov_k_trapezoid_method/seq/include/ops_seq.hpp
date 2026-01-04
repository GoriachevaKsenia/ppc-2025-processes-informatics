#pragma once

#include "lifanov_k_trapezoid_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lifanov_k_trapezoid_method {

class LifanovKTrapezoidMethodSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LifanovKTrapezoidMethodSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace lifanov_k_trapezoid_method
