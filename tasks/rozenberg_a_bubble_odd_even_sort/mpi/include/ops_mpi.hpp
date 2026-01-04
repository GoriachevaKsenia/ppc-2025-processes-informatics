#pragma once

#include "rozenberg_a_bubble_odd_even_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rozenberg_a_bubble_odd_even_sort {

class RozenbergABubbleOddEvenSortMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit RozenbergABubbleOddEvenSortMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void LocalBubbleSort(InType &local_buf);
  static void ExchangeAndMerge(InType &local_buf, int neighbor, int chunk, int neighbor_n, int rank);
};

}  // namespace rozenberg_a_bubble_odd_even_sort
