#pragma once

#include "kondakov_v_reduce/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kondakov_v_reduce {

class KondakovVReduceTaskSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit KondakovVReduceTaskSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kondakov_v_reduce
