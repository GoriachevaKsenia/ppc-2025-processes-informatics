#pragma once

#include "sakharov_a_transmission_from_one_to_all/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sakharov_a_transmission_from_one_to_all {

class SakharovATransmissionFromOneToAllSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SakharovATransmissionFromOneToAllSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace sakharov_a_transmission_from_one_to_all
