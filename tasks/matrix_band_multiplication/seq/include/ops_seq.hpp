#pragma once

#include <vector>

#include "matrix_band_multiplication/common/include/common.hpp"
#include "task/include/task.hpp"

namespace matrix_band_multiplication {

class MatrixBandMultiplicationSeq : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit MatrixBandMultiplicationSeq(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> result_;
  std::vector<double> transposed_b_;
};

}  // namespace matrix_band_multiplication
