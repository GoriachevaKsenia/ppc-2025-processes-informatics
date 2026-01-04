#pragma once

#include "lobanov_d_multiply_matrix_ccs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lobanov_d_multiply_matrix_ccs {
class LobanovDMultiplyMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LobanovDMultiplyMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void ComputeTransposedMatrix(const CompressedColumnMatrix &source_matrix,
                                      CompressedColumnMatrix &transposed_result);
  static void PerformMatrixMultiplication(const CompressedColumnMatrix &first_matrix,
                                          const CompressedColumnMatrix &second_matrix,
                                          CompressedColumnMatrix &product_result);
};

}  // namespace lobanov_d_multiply_matrix_ccs
