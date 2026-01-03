#pragma once

#include "goriacheva_k_strassen_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace goriacheva_k_strassen_algorithm {

class GoriachevaKStrassenAlgorithmMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit GoriachevaKStrassenAlgorithmMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  InType input_matrices_;
  OutType result_matrix_;

  Matrix MpiStrassenTop(const Matrix &a, const Matrix &b);
};

}  // namespace goriacheva_k_strassen_algorithm
