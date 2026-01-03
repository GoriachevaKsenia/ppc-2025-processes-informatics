#pragma once

#include "dolov_v_torus_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dolov_v_torus_topology {

class DolovVTorusTopologySEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit DolovVTorusTopologySEQ(InType in);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  InType internal_input_;
  OutType internal_output_;
};

}  // namespace dolov_v_torus_topology
