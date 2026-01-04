#pragma once

#include "gasenin_l_mult_int_mstep_trapez/common/include/common.hpp"
#include "task/include/task.hpp"

namespace gasenin_l_mult_int_mstep_trapez {

class GaseninLMultIntMstepTrapezSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit GaseninLMultIntMstepTrapezSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace gasenin_l_mult_int_mstep_trapez
