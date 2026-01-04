#pragma once

#include "nikitin_a_buble_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nikitin_a_buble_sort {

class NikitinABubleSortSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit NikitinABubleSortSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace nikitin_a_buble_sort
