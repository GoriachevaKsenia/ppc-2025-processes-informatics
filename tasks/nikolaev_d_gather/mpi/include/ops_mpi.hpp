#pragma once

#include "nikolaev_d_gather/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nikolaev_d_gather {

class NikolaevDGatherMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit NikolaevDGatherMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace nikolaev_d_gather
