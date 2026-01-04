#pragma once

#include "shkrebko_m_hypercube/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shkrebko_m_hypercube {

class ShkrebkoMHypercubeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ShkrebkoMHypercubeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace shkrebko_m_hypercube
