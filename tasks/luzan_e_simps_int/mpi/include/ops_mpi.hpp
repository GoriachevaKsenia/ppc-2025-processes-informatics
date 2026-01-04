#pragma once

#include "luzan_e_simps_int/common/include/common.hpp"
#include "task/include/task.hpp"

namespace luzan_e_simps_int {

class LuzanESimpsIntMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LuzanESimpsIntMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace luzan_e_simps_int
