#pragma once

#include "gonozov_l_global_search/common/include/common.hpp"
#include "task/include/task.hpp"

namespace gonozov_l_global_search {

class GonozovLGlobalSearchMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit GonozovLGlobalSearchMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace gonozov_l_global_search
