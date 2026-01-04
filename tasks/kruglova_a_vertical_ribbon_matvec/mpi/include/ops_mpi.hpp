#pragma once

#include "kruglova_a_vertical_ribbon_matvec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kruglova_a_vertical_ribbon_matvec {

class KruglovaAVerticalRibbMatMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KruglovaAVerticalRibbMatMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kruglova_a_vertical_ribbon_matvec
