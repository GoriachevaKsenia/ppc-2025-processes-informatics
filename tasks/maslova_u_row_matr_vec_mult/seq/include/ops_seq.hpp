#pragma once

#include "maslova_u_row_matr_vec_mult/common/include/common.hpp"
#include "task/include/task.hpp"

namespace maslova_u_row_matr_vec_mult {

class MaslovaURowMatrVecMultSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit MaslovaURowMatrVecMultSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace maslova_u_row_matr_vec_mult
