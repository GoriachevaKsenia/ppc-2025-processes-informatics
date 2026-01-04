#pragma once

#include "task/include/task.hpp"
#include "votincev_d_qsort_batcher/common/include/common.hpp"

namespace votincev_d_qsort_batcher {

class VotincevDQsortBatcherSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VotincevDQsortBatcherSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // ==============================
  // мои дополнительные функции ===
  static int Partition(double *arr, int l, int h);
  static void QuickSort(double *arr, int left, int right);
};

}  // namespace votincev_d_qsort_batcher
