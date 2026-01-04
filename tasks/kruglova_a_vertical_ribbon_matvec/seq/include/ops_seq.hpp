#pragma once

#include "kruglova_a_vertical_ribbon_matvec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kruglova_a_vertical_ribbon_matvec {

class KruglovaAVerticalRibbMatSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KruglovaAVerticalRibbMatSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kruglova_a_vertical_ribbon_matvec
