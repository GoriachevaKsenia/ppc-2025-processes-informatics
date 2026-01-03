#pragma once

#include "goriacheva_k_reduce/common/include/common.hpp"
#include "task/include/task.hpp"

namespace goriacheva_k_reduce {

class GoriachevaKReduceSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit GoriachevaKReduceSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace goriacheva_k_reduce
