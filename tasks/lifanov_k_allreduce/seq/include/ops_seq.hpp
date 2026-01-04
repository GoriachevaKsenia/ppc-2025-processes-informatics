#pragma once

#include "lifanov_k_allreduce/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lifanov_k_allreduce {

class LifanovKAllreduceSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LifanovKAllreduceSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace lifanov_k_allreduce
