#pragma once

#include "luzan_e_simps_int/common/include/common.hpp"
#include "task/include/task.hpp"

namespace luzan_e_simps_int {

class LuzanESimpsIntSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LuzanESimpsIntSEQ(const InType &in);

 private:
  bool ValidationImpl() override;      //= input data check
  bool PreProcessingImpl() override;   //= pre proc
  bool RunImpl() override;             //= PARALLEL
  bool PostProcessingImpl() override;  //= back to reality
};

}  // namespace luzan_e_simps_int
