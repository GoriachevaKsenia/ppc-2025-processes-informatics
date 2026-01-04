#pragma once

#include <vector>

#include "ashihmin_d_calculate_integrals_by_simpson/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ashihmin_d_calculate_integrals_by_simpson {

class AshihminDCalculateIntegralsBySimpsonMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit AshihminDCalculateIntegralsBySimpsonMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static double IntegrandFunction(const std::vector<double> &coordinates);
};

}  // namespace ashihmin_d_calculate_integrals_by_simpson
