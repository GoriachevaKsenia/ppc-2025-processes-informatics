#pragma once

#include "maslova_u_fast_sort_simple/common/include/common.hpp"
#include "task/include/task.hpp"

namespace maslova_u_fast_sort_simple {

class MaslovaUFastSortSimpleSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit MaslovaUFastSortSimpleSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace maslova_u_fast_sort_simple
