#pragma once

#include <vector>

#include "goriacheva_k_strassen_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace goriacheva_k_strassen_algorithm {

class GoriachevaKStrassenAlgorithmSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit GoriachevaKStrassenAlgorithmSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  InType input_matrices_;
  OutType result_matrix_;
};

}  // namespace goriacheva_k_strassen_algorithm
