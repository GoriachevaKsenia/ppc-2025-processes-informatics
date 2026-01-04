#pragma once

#include "sannikov_i_shtrassen_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sannikov_i_shtrassen_algorithm {

class SannikovIShtrassenAlgorithmMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SannikovIShtrassenAlgorithmMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace sannikov_i_shtrassen_algorithm
