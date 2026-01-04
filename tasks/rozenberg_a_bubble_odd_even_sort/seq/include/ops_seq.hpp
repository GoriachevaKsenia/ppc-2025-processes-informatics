#pragma once

#include "rozenberg_a_bubble_odd_even_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rozenberg_a_bubble_odd_even_sort {

class RozenbergABubbleOddEvenSortSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit RozenbergABubbleOddEvenSortSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace rozenberg_a_bubble_odd_even_sort
