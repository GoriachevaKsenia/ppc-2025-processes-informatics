#pragma once

#include "kutuzov_i_torus_grid/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kutuzov_i_torus_grid {

class KutuzovIThorusGridMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KutuzovIThorusGridMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static int GetNextStep(int current_x, int current_y, int dest_x, int dest_y, int columns, int rows);

  static void GenerateTopology(int &columns, int &rows, int process_count);
};

}  // namespace kutuzov_i_torus_grid
