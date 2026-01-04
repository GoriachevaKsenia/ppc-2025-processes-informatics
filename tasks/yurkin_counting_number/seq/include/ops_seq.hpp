#pragma once

#include "task/include/task.hpp"
#include "yurkin_counting_number/common/include/common.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit YurkinCountingNumberSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace yurkin_counting_number
