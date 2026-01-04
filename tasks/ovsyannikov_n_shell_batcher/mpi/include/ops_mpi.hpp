#pragma once

#include <vector>

#include "ovsyannikov_n_shell_batcher/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ovsyannikov_n_shell_batcher {

class OvsyannikovNShellBatcherMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit OvsyannikovNShellBatcherMPI(const InType &in) {
    SetTypeOfTask(GetStaticTypeOfTask());
    GetInput() = in;
    GetOutput() = {};
  }

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> local_;
  std::vector<int> counts_;
  std::vector<int> displs_;
  int world_rank_{0};
  int world_size_{1};
};

}  // namespace ovsyannikov_n_shell_batcher
