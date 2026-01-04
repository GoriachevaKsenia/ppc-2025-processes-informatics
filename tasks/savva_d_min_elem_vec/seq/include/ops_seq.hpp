#pragma once

#include "savva_d_min_elem_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace savva_d_min_elem_vec {

class SavvaDMinElemVecSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SavvaDMinElemVecSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace savva_d_min_elem_vec
