#pragma once

#include "sannikov_i_shtrassen_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sannikov_i_shtrassen_algorithm {

class SannikovIShtrassenAlgorithmSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SannikovIShtrassenAlgorithmSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace sannikov_i_shtrassen_algorithm
