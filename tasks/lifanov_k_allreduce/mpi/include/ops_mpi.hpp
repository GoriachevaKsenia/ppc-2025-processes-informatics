#pragma once

#include <vector>

#include "lifanov_k_allreduce/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lifanov_k_allreduce {

class LifanovKAllReduceMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LifanovKAllReduceMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> local_input_;
};

}  // namespace lifanov_k_allreduce
