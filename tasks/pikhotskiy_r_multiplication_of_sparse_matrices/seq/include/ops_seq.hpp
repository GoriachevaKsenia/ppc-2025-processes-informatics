#pragma once

#include "pikhotskiy_r_multiplication_of_sparse_matrices/common/include/common.hpp"
#include "task/include/task.hpp"

namespace pikhotskiy_r_multiplication_of_sparse_matrices {

class SparseMatrixMultiplicationSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SparseMatrixMultiplicationSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  SparseMatrixCRS mat_a_;
  SparseMatrixCRS mat_b_transposed_;
};

}  // namespace pikhotskiy_r_multiplication_of_sparse_matrices
