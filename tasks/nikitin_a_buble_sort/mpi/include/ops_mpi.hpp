#pragma once

#include <vector>

#include "nikitin_a_buble_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nikitin_a_buble_sort {

class NikitinABubleSortMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit NikitinABubleSortMPI(const InType &in);

 private:
  std::vector<double> data_;
  int n_ = 0;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void LocalSort(std::vector<double> &local, int global_offset, int phase_parity);
  static void ExchangeRight(std::vector<double> &local, const std::vector<int> &counts, const std::vector<int> &displs,
                            int rank, int comm_size, int phase_parity, int tag);
  static void ExchangeLeft(std::vector<double> &local, const std::vector<int> &counts, const std::vector<int> &displs,
                           int rank, int phase_parity, int tag);
};

}  // namespace nikitin_a_buble_sort
