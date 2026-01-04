#pragma once

#include <vector>

#include "goriacheva_k_violation_order_elem_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace goriacheva_k_violation_order_elem_vec {

class GoriachevaKViolationOrderElemVecMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit GoriachevaKViolationOrderElemVecMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void ScatterInput(int rank, int size, int n, std::vector<int> &local) const;

  [[nodiscard]] static int CountLocalViolations(const std::vector<int> &local);
  [[nodiscard]] static int CheckBoundaryViolation(int rank, int size, const std::vector<int> &local);

  std::vector<int> input_vec_;
  int result_ = 0;
};

}  // namespace goriacheva_k_violation_order_elem_vec
