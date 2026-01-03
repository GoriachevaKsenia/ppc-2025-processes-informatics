#pragma once

#include "morozova_s_matrix_max_value/common/include/common.hpp"
#include "task/include/task.hpp"

namespace morozova_s_matrix_max_value {

class MorozovaSMatrixMaxValueMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit MorozovaSMatrixMaxValueMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace morozova_s_matrix_max_value
