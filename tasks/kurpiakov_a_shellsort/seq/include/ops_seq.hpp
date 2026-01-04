#pragma once

#include "kurpiakov_a_shellsort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kurpiakov_a_shellsort {

class KurpiakovAShellsortSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KurpiakovAShellsortSEQ(const InType &in);

 private:
  OutType data_;
  OutType gaps_;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kurpiakov_a_shellsort
