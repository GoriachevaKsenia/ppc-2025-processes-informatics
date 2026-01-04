#pragma once

#include "luzan_e_matrix_horis_rib_mult_sheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace luzan_e_matrix_horis_rib_mult_sheme {

class LuzanEMatrixHorisRibMultShemeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LuzanEMatrixHorisRibMultShemeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;      //= input data check
  bool PreProcessingImpl() override;   //= pre proc
  bool RunImpl() override;             //= PARALLEL
  bool PostProcessingImpl() override;  //= back to reality
};

}  // namespace luzan_e_matrix_horis_rib_mult_sheme
