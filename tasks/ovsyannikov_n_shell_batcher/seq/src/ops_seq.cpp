#include "ovsyannikov_n_shell_batcher/seq/include/ops_seq.hpp"

#include "ovsyannikov_n_shell_batcher/common/include/add_functs.hpp"
#include "ovsyannikov_n_shell_batcher/common/include/common.hpp"

namespace ovsyannikov_n_shell_batcher {

OvsyannikovNShellBatcherSEQ::OvsyannikovNShellBatcherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool OvsyannikovNShellBatcherSEQ::ValidationImpl() {
  return true;
}

bool OvsyannikovNShellBatcherSEQ::PreProcessingImpl() {
  data_ = GetInput();
  GetOutput().clear();
  return true;
}

bool OvsyannikovNShellBatcherSEQ::RunImpl() {
  ShellSort(&data_);
  GetOutput() = data_;
  return true;
}

bool OvsyannikovNShellBatcherSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace ovsyannikov_n_shell_batcher
