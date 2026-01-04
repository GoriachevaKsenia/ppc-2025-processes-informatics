#pragma once

#include "konstantinov_s_elem_vec_sign_change_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace konstantinov_s_elem_vec_sign_change_count {

class KonstantinovSElemVecSignChangeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KonstantinovSElemVecSignChangeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  static void CountSignChange(int &res, const EType *data, int start, int iterations);
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace konstantinov_s_elem_vec_sign_change_count
