#pragma once

#include "kutuzov_i_simpson_integration/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kutuzov_i_simpson_integration {

class KutuzovISimpsonIntegrationMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KutuzovISimpsonIntegrationMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static double GetWeight(int i, int n);
};

}  // namespace kutuzov_i_simpson_integration
