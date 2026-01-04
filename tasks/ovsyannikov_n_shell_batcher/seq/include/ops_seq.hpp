#pragma once

#include "ovsyannikov_n_shell_batcher/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ovsyannikov_n_shell_batcher {

class OvsyannikovNShellBatcherSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit OvsyannikovNShellBatcherSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  InType data_;
};

}  // namespace ovsyannikov_n_shell_batcher
