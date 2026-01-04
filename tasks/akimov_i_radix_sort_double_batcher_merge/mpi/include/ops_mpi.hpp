#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "akimov_i_radix_sort_double_batcher_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace akimov_i_radix_sort_double_batcher_merge {

class AkimovIRadixBatcherSortMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit AkimovIRadixBatcherSortMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Radix helpers
  static uint64_t PackDouble(double v) noexcept;
  static double UnpackDouble(uint64_t k) noexcept;
  static void LsdRadixSort(std::vector<double> &arr);

  // Batcher / merge-network helpers
  static void CmpSwap(std::vector<double> &arr, int i, int j) noexcept;
  // NOLINTNEXTLINE(misc-no-recursion)
  static void OddEvenMergeRec(std::vector<double> &arr, int start, int len, int stride);
  // NOLINTNEXTLINE(misc-no-recursion)
  static void OddEvenMergeSortRec(std::vector<double> &arr, int start, int len);

  static std::vector<std::pair<int, int>> BuildOddEvenPhasePairs(int procs);
  static void ExchangeAndSelect(std::vector<double> &local, int partner, int rank, bool keep_lower);

  // MPI data distribution helpers
  static void ComputeCountsDispls(int total, int world, std::vector<int> &counts, std::vector<int> &displs);
  static void ScatterData(int total, int world, int rank, std::vector<double> &data, std::vector<int> &counts,
                          std::vector<int> &displs, std::vector<double> &local);
  static void GatherData(std::vector<double> &local_data, int total_size, int world_size, int rank,
                         std::vector<double> &sorted_data);
  static void PerformNetworkMerge(std::vector<double> &local, int world, int rank);
};

}  // namespace akimov_i_radix_sort_double_batcher_merge
