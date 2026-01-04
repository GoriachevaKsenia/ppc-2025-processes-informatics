#pragma once

#include "../../common/include/common.hpp"
#include "task/include/task.hpp"

namespace shekhirev_v_custom_reduce_seq {

class CustomReduceSequential : public shekhirev_v_custom_reduce::BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit CustomReduceSequential(const shekhirev_v_custom_reduce::InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace shekhirev_v_custom_reduce_seq
