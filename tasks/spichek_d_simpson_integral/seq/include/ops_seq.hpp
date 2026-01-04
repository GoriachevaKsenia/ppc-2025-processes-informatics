#pragma once

#include "spichek_d_simpson_integral/common/include/common.hpp"
#include "task/include/task.hpp"

namespace spichek_d_simpson_integral {

class SpichekDSimpsonIntegralSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SpichekDSimpsonIntegralSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace spichek_d_simpson_integral
