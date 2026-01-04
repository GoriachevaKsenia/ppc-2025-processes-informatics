#pragma once

#include <vector>

#include "ashihmin_d_scatter_trans_from_one_to_all/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ashihmin_d_scatter_trans_from_one_to_all {

template <typename T>
class AshihminDScatterTransFromOneToAllMPI : public BaseTask<T> {
 public:
  using InType = ScatterParams;
  using OutType = std::vector<T>;

  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit AshihminDScatterTransFromOneToAllMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace ashihmin_d_scatter_trans_from_one_to_all
