#pragma once

#include <vector>

#include "goriacheva_k_reduce/common/include/common.hpp"
#include "task/include/task.hpp"

namespace goriacheva_k_reduce {

class GoriachevaKReduceMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit GoriachevaKReduceMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> local_input_;
};

}  // namespace goriacheva_k_reduce
