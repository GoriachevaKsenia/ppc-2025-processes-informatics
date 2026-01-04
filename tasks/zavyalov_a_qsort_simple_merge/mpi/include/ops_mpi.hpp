#pragma once

#include "task/include/task.hpp"
#include "zavyalov_a_qsort_simple_merge/common/include/common.hpp"

namespace zavyalov_a_qsort_simple_merge {

class ZavyalovAQsortMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ZavyalovAQsortMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zavyalov_a_qsort_simple_merge
