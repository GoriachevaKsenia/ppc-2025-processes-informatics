#pragma once

#include "gusev_d_star/common/include/common.hpp"
#include "task/include/task.hpp"

namespace gusev_d_star {

class GusevDStarMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit GusevDStarMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void RunAsMaster(int size, int input);
  static void RunAsWorker(int input);
};

}  // namespace gusev_d_star
