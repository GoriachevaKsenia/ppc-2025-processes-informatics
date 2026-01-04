#pragma once

#include <utility>
#include <vector>

#include "nikitin_a_fox_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nikitin_a_fox_algorithm {

class NikitinAFoxAlgorithmMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit NikitinAFoxAlgorithmMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Вспомогательные методы для уменьшения когнитивной сложности
  static bool ValidateMatricesOnRoot(const std::vector<std::vector<double>> &matrix_a,
                                     const std::vector<std::vector<double>> &matrix_b);
  void DistributeMatrixB(int n, std::vector<double> &local_b);
  void SendMatrixAToProcess(int dest, int rows_per_proc, int remainder, int n, int &current_row);
  void DistributeMatrixA(int rank, int size, int n, int local_rows, std::vector<double> &local_a);
  static void LocalMatrixMultiply(int n, int local_rows, const std::vector<double> &local_a,
                                  const std::vector<double> &local_b, std::vector<double> &local_c);
  void CopyLocalResultsToOutput(int local_rows, int n, const std::vector<double> &local_c);
  void ReceiveResultsFromProcess(int src, int rows_per_proc, int remainder, int n, int &current_row);
  void GatherResults(int rank, int size, int n, int rows_per_proc, int remainder, int local_rows,
                     const std::vector<double> &local_c);
  void CreateAndSendFlatResult(int n);
  void ReceiveFlatResultAndCreateMatrix(int n);
  void BroadcastResultToAll(int rank, int n);
  int GetMatrixSize(int rank);
  static std::pair<int, int> CalculateRowDistribution(int n, int size, int rank);
};

}  // namespace nikitin_a_fox_algorithm
