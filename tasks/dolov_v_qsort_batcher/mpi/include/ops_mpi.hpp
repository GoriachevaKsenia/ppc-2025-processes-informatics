#pragma once

#include <vector>

#include "dolov_v_qsort_batcher/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dolov_v_qsort_batcher {

class DolovVQsortBatcherMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit DolovVQsortBatcherMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void FastSort(double *data, int low, int high);
  static int GetSplitIndex(double *data, int low, int high);

  void DistributeData(int world_rank, int world_size);
  void CollectData(int world_rank, int world_size);

  void ExecuteBatcherParallel(int world_rank, int world_size);
  void BatcherStep(int p_step, int k, int j, int i, int world_rank, int world_size);
  static void MergeSequences(const std::vector<double> &first, const std::vector<double> &second,
                             std::vector<double> &result, bool take_low);

  std::vector<double> local_buffer_;
  std::vector<int> part_sizes_;
  std::vector<int> part_offsets_;
  int total_count_ = 0;
};

}  // namespace dolov_v_qsort_batcher
