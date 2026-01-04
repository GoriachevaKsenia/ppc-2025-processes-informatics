#pragma once

#include "krykov_e_simple_iterations/common/include/common.hpp"
#include "task/include/task.hpp"

namespace krykov_e_simple_iterations {

class KrykovESimpleIterationsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KrykovESimpleIterationsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace krykov_e_simple_iterations
