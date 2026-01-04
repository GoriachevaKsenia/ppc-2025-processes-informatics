#pragma once

#include "spichek_d_jacobi/common/include/common.hpp"
#include "task/include/task.hpp"

namespace spichek_d_jacobi {

class SpichekDJacobiMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  // ИСПРАВЛЕНО: имя параметра изменено с task_data на in, чтобы совпадать с .cpp
  explicit SpichekDJacobiMPI(InType in);

 private:
  InType input_;
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace spichek_d_jacobi
