#pragma once

#include "kondakov_v_reduce/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kondakov_v_reduce {

class KondakovVReduceTaskMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit KondakovVReduceTaskMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kondakov_v_reduce
