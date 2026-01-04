#pragma once

#include "gutyansky_a_matrix_band_multiplication/common/include/common.hpp"
#include "task/include/task.hpp"

namespace gutyansky_a_matrix_band_multiplication {

class GutyanskyAMatrixBandMultiplicationSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit GutyanskyAMatrixBandMultiplicationSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace gutyansky_a_matrix_band_multiplication
