#pragma once

#include "liulin_y_vert_strip_diag_matrix_vect_mult/common/include/common.hpp"
#include "task/include/task.hpp"

namespace liulin_y_vert_strip_diag_matrix_vect_mult {

class LiulinYVertStripDiagMatrixVectMultMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LiulinYVertStripDiagMatrixVectMultMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace liulin_y_vert_strip_diag_matrix_vect_mult
