#pragma once

#include "pikhotskiy_r_multiplication_of_sparse_matrices/common/include/common.hpp"
#include "task/include/task.hpp"

namespace pikhotskiy_r_multiplication_of_sparse_matrices {

class SparseMatrixMultiplicationMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SparseMatrixMultiplicationMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void BroadcastSparseMatrix(SparseMatrixCRS &matrix, int root);
  void GatherResults(const SparseMatrixCRS &local_result, int my_num_rows);

  SparseMatrixCRS mat_a_;
  SparseMatrixCRS mat_b_;
  SparseMatrixCRS mat_b_transposed_;
};

}  // namespace pikhotskiy_r_multiplication_of_sparse_matrices
