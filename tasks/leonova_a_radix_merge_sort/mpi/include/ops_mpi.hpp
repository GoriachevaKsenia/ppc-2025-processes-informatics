#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "leonova_a_radix_merge_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace leonova_a_radix_merge_sort {

class LeonovaARadixMergeSortMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LeonovaARadixMergeSortMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void HierarchicalMerge();
  void BroadcastResult();
  static void RadixMergeSort(std::vector<double> &arr, size_t left, size_t right);
  static void RadixSort(std::vector<double> &arr, size_t left, size_t right);

  static uint64_t TransformDoubleToKey(double value);
  static void ComputeKeysForVector(const std::vector<double> &vec, std::vector<uint64_t> &keys);
  static void MergeWithKeys(std::vector<double> &result, const std::vector<double> &vec1,
                            const std::vector<uint64_t> &keys1, const std::vector<double> &vec2,
                            const std::vector<uint64_t> &keys2);

  bool ShouldMergeWithPartner(int step, int &partner_rank, bool &is_sender) const;
  void SendLocalData(int partner_rank);
  void ReceiveAndMergeData(int partner_rank, std::vector<double> &partner_data_buffer,
                           std::vector<uint64_t> &partner_keys_buffer);

  static void CountFrequencies(const std::vector<uint64_t> &keys, int shift, std::array<size_t, 256> &count);
  static void ComputePrefixSums(std::array<size_t, 256> &count);
  static void DistributeElements(const std::vector<double> &arr, const std::vector<uint64_t> &keys, size_t left,
                                 int shift, const std::array<size_t, 256> &count, std::vector<double> &temp_arr,
                                 std::vector<uint64_t> &temp_keys);
  static void MergeTwoParts(const std::vector<double> &arr, size_t left, size_t mid, size_t right,
                            std::vector<double> &merged);
  static void CopyRemainingElements(const std::vector<double> &arr, size_t src_offset, size_t src_size,
                                    std::vector<double> &merged, size_t dest_offset);

  // Данные процесса
  std::vector<double> local_data_;
  std::vector<uint64_t> local_keys_;
  int rank_{};
  int world_size_{};

  // Константы
  static constexpr size_t kRadixThreshold = 32;
  static constexpr int kByteSize = 8;
  static constexpr int kNumBytes = 8;
  static constexpr int kNumCounters = 256;
};

}  // namespace leonova_a_radix_merge_sort
