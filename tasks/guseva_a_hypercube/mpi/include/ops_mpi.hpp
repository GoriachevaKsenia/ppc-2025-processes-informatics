#pragma once

#include <memory>

#include "guseva_a_hypercube/common/include/common.hpp"
#include "guseva_a_hypercube/mpi/include/hypercube.hpp"
#include "task/include/task.hpp"

namespace guseva_a_hypercube {

class GusevaAHypercubeMine : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit GusevaAHypercubeMine(const InType &in);

 private:
  std::unique_ptr<Hypercube> hypercube_handler_;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace guseva_a_hypercube
