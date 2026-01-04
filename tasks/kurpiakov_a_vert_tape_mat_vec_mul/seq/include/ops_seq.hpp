#pragma once

#include "kurpiakov_a_vert_tape_mat_vec_mul/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kurpiakov_a_vert_tape_mat_vec_mul {

class KurpiakovAVretTapeMulSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KurpiakovAVretTapeMulSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kurpiakov_a_vert_tape_mat_vec_mul
