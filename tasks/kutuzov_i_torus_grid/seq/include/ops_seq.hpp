#pragma once

#include <vector>

#include "kutuzov_i_torus_grid/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kutuzov_i_torus_grid {

class KutuzovIThorusGridSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KutuzovIThorusGridSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> dummy_route_ = {2, 0, 2, 6};
};
}  // namespace kutuzov_i_torus_grid
