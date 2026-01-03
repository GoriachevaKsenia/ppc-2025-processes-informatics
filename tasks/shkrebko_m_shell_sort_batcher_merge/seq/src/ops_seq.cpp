#include "shkrebko_m_shell_sort_batcher_merge/seq/include/ops_seq.hpp"

#include <vector>

#include "shkrebko_m_shell_sort_batcher_merge/common/include/common.hpp"
#include "shkrebko_m_shell_sort_batcher_merge/common/include/utils.hpp"

namespace shkrebko_m_shell_sort_batcher_merge {

ShkrebkoMShellSortBatcherMergeSEQ::ShkrebkoMShellSortBatcherMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool ShkrebkoMShellSortBatcherMergeSEQ::ValidationImpl() {
  return true;
}

bool ShkrebkoMShellSortBatcherMergeSEQ::PreProcessingImpl() {
  data_ = GetInput();
  GetOutput().clear();
  return true;
}

bool ShkrebkoMShellSortBatcherMergeSEQ::RunImpl() {
  ShellSort(&data_);
  GetOutput() = data_;
  return true;
}

bool ShkrebkoMShellSortBatcherMergeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shkrebko_m_shell_sort_batcher_merge
