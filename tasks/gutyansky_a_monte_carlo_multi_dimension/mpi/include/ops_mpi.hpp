#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gutyansky_a_monte_carlo_multi_dimension/common/include/common.hpp"
#include "gutyansky_a_monte_carlo_multi_dimension/common/include/integration_task.hpp"
#include "task/include/task.hpp"

namespace gutyansky_a_monte_carlo_multi_dimension {

class GutyanskyAMonteCarloMultiDimensionMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit GutyanskyAMonteCarloMultiDimensionMPI(const InType &in);

 private:
  static int ComputePackedTaskSize(size_t n_dims);
  static void PackTaskData(const IntegrationTask &task, std::vector<uint8_t> &buffer);
  static void UnpackTaskData(const std::vector<uint8_t> &buffer, IntegrationTask &task);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace gutyansky_a_monte_carlo_multi_dimension
