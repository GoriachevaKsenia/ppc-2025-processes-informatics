#pragma once

#include <cstdint>
#include <vector>

#include "akimov_i_radix_sort_double_batcher_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace akimov_i_radix_sort_double_batcher_merge {

class AkimovIRadixBatcherSortSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit AkimovIRadixBatcherSortSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static uint64_t PackDouble(double v) noexcept;
  static double UnpackDouble(uint64_t k) noexcept;
  static void LsdRadixSort(std::vector<double> &arr);
};
}  // namespace akimov_i_radix_sort_double_batcher_merge
