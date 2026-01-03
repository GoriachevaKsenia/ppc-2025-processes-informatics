#pragma once

#include "dolov_v_monte_carlo_integration/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dolov_v_monte_carlo_integration {

class DolovVMonteCarloIntegrationMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit DolovVMonteCarloIntegrationMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace dolov_v_monte_carlo_integration
