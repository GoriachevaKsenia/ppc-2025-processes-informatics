#pragma once

#include <vector>

#include "mityaeva_d_striped_horizontal_matrix_vector/common/include/common.hpp"
#include "task/include/task.hpp"

namespace mityaeva_d_striped_horizontal_matrix_vector {

class StripedHorizontalMatrixVectorMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit StripedHorizontalMatrixVectorMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

std::vector<double> ProcessLocalRows(const std::vector<double> &input, int start_row, int my_rows, int cols,
                                     int total_rows);
void GatherResults(int rank, int size, int rows, int rows_per_process, int remainder,
                   const std::vector<double> &local_result, std::vector<double> &output);

}  // namespace mityaeva_d_striped_horizontal_matrix_vector
