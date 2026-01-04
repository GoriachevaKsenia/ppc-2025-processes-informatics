#pragma once

#include "mityaeva_d_striped_horizontal_matrix_vector/common/include/common.hpp"
#include "task/include/task.hpp"

namespace mityaeva_d_striped_horizontal_matrix_vector {

class StripedHorizontalMatrixVectorSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit StripedHorizontalMatrixVectorSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int rows_{0};
  int cols_{0};
};

}  // namespace mityaeva_d_striped_horizontal_matrix_vector
