#pragma once

#include <vector>

#include "frolova_s_sum_elem_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace frolova_s_sum_elem_matrix {

class FrolovaSSumElemMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit FrolovaSSumElemMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static double ProcessMaster(int process_n, const std::vector<double> &vect_data);
  static void ProcessWorker();
};

}  // namespace frolova_s_sum_elem_matrix
