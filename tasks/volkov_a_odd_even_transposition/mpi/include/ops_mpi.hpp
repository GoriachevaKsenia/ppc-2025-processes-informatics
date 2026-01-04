#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "volkov_a_odd_even_transposition/common/include/common.hpp"

namespace volkov_a_odd_even_transposition {

class OddEvenSortMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit OddEvenSortMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void CalculateDistribution(int n, int size, std::vector<int> &counts, std::vector<int> &displs);
  static void PerformCompareSplit(InType &local_data, int partner_rank, int my_rank);
  static int GetNeighbor(int phase, int rank, int size);
};

}  // namespace volkov_a_odd_even_transposition
