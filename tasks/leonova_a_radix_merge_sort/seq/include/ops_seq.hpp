#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "leonova_a_radix_merge_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace leonova_a_radix_merge_sort {

class LeonovaARadixMergeSortSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LeonovaARadixMergeSortSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void RadixMergeSort(std::vector<double> &arr, size_t left, size_t right);
  static void RadixSort(std::vector<double> &arr, size_t left, size_t right);
  static void SimpleRadixMerge(std::vector<double> &arr, size_t left, size_t mid, size_t right);

  static uint64_t TransformDoubleToKey(double value);

  // Константы
  static constexpr size_t kRadixThreshold = 32;  // Порог для переключения на RadixSort
  static constexpr int kByteSize = 8;
  static constexpr int kNumBytes = 8;
  static constexpr int kNumCounters = 256;
};

}  // namespace leonova_a_radix_merge_sort
