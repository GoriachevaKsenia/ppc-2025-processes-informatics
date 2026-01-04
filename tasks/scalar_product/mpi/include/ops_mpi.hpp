#pragma once

#include <vector>

#include "scalar_product/common/include/common.hpp"
#include "task/include/task.hpp"

namespace scalar_product {

class ScalarProductMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ScalarProductMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int rank_ = 0;
  int world_size_ = 0;
  int local_sum_ = 0;
  int result_ = 0;
  std::vector<int> local_vector_a_;
  std::vector<int> local_vector_b_;
};

}  // namespace scalar_product
