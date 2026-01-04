#pragma once

#include <utility>
#include <vector>

#include "leonova_a_star/common/include/common.hpp"
#include "task/include/task.hpp"

namespace leonova_a_star {

class LeonovaAStarMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LeonovaAStarMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  bool ValidateMatricesOnMaster();

  static std::vector<std::vector<int>> MultiplyMatricesMpi(const std::vector<std::vector<int>> &matrix_a,
                                                           const std::vector<std::vector<int>> &matrix_b);

  void BroadcastResult(int rank);

  bool ResizeOutputMatrix(int rows, int cols);
  std::pair<int, int> GetResultDimensions(int rank);
  static bool ValidateDimensions(int rows, int cols);
  void BroadcastFromMaster(int rows, int cols);
  void ReceiveFromMaster(int rows, int cols);
};

}  // namespace leonova_a_star
