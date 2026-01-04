#pragma once

#include <vector>

#include "maslova_u_fast_sort_simple/common/include/common.hpp"
#include "task/include/task.hpp"

namespace maslova_u_fast_sort_simple {

class MaslovaUFastSortSimpleMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit MaslovaUFastSortSimpleMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void TreeMerge(std::vector<int> &local_vec, int rank, int size);
};

}  // namespace maslova_u_fast_sort_simple
