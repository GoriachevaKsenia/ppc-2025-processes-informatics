#pragma once

#include <utility>
#include <vector>

#include "chyokotov_a_seidel_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chyokotov_a_seidel_method {

class ChyokotovASeidelMethodMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ChyokotovASeidelMethodMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  bool CheckMatrix();
  void ExchangeMatrixData(int rank, int size, int n, std::vector<std::vector<double>> &a, std::vector<double> &b,
                          const std::vector<int> &displs, int local_rows);
  std::pair<int, int> DistributeMatrixData(int rank, int size, int n, std::vector<std::vector<double>> &a,
                                           std::vector<double> &b, std::vector<int> &displs, std::vector<int> &counts);
  void Iteration(int n, int local_rows, int local_start, std::vector<std::vector<double>> &my_a,
                 std::vector<double> &my_b, std::vector<int> &displs, std::vector<int> &counts);
};

}  // namespace chyokotov_a_seidel_method
