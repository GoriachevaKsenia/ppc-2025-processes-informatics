#pragma once

#include "kurpiakov_a_shellsort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kurpiakov_a_shellsort {

class KurpiakovAShellsortMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KurpiakovAShellsortMPI(const InType &in);

 private:
  OutType data_;
  int world_size_ = 0;
  int rank_ = 0;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void ShellSortLocal(OutType &arr);
  void MergeSortedArrays(const OutType &send_counts, const OutType &gathered_data, const OutType &displs, const int &n);
};

}  // namespace kurpiakov_a_shellsort
