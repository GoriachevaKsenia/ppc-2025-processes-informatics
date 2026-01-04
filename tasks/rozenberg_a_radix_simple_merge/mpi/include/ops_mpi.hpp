#pragma once

#include "rozenberg_a_radix_simple_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rozenberg_a_radix_simple_merge {

class RozenbergARadixSimpleMergeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit RozenbergARadixSimpleMergeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void LocalRadixSort(InType &data);
  static void ExchangeAndMerge(InType &local_buf, int rank, int size);
};

}  // namespace rozenberg_a_radix_simple_merge
