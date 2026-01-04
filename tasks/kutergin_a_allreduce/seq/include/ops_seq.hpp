#pragma once

#include "../../common/include/common.hpp"
#include "task/include/task.hpp"

namespace kutergin_a_allreduce {

class AllreduceSequential : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit AllreduceSequential(const InType &in);

 protected:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kutergin_a_allreduce
