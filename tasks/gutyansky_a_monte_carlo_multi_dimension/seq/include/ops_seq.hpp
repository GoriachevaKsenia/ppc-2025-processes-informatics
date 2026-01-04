#pragma once

#include "gutyansky_a_monte_carlo_multi_dimension/common/include/common.hpp"
#include "task/include/task.hpp"

namespace gutyansky_a_monte_carlo_multi_dimension {

class GutyanskyAMonteCarloMultiDimensionSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit GutyanskyAMonteCarloMultiDimensionSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace gutyansky_a_monte_carlo_multi_dimension
