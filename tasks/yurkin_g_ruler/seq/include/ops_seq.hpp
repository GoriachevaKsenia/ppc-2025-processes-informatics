#pragma once

#include "task/include/task.hpp"
#include "yurkin_g_ruler/common/include/common.hpp"

namespace yurkin_g_ruler {

class YurkinGRulerSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit YurkinGRulerSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace yurkin_g_ruler
