#pragma once

#include "sakharov_a_cannon_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sakharov_a_cannon_algorithm {

class SakharovACannonAlgorithmMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SakharovACannonAlgorithmMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace sakharov_a_cannon_algorithm
