#pragma once

#include "shkrebko_m_shell_sort_batcher_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shkrebko_m_shell_sort_batcher_merge {

class ShkrebkoMShellSortBatcherMergeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit ShkrebkoMShellSortBatcherMergeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  InType data_;
};

}  // namespace shkrebko_m_shell_sort_batcher_merge
