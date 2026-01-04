#pragma once

#include "task/include/task.hpp"
#include "volkov_a_odd_even_transposition/common/include/common.hpp"

namespace volkov_a_odd_even_transposition {

class OddEvenSortSeq : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit OddEvenSortSeq(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace volkov_a_odd_even_transposition
