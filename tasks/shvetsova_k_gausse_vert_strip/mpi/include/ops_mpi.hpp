#pragma once

#include <vector>

#include "shvetsova_k_gausse_vert_strip/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shvetsova_k_gausse_vert_strip {

class ShvetsovaKGaussVertStripMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ShvetsovaKGaussVertStripMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  InType input_data_;
  int size_of_rib_ = 0;

  static int GetOwnerOfColumn(int k, int n, int size);
  static int GetColumnStartIndex(int rank, int n, int size);
  static int GetColumnEndIndex(int rank, int n, int size);
  static int CalculateRibWidth(const std::vector<std::vector<double>> &matrix, int n);
  void ForwardElimination(int n, int rank, int size, int c0, int local_cols, std::vector<std::vector<double>> &a_local,
                          std::vector<double> &b) const;
  [[nodiscard]] std::vector<double> BackSubstitution(int n, int rank, int size, int c0,
                                                     const std::vector<std::vector<double>> &a_local,
                                                     const std::vector<double> &b) const;
  static void ScatterColumns(int n, int rank, int size, int c0, int local_cols,
                             const std::vector<std::vector<double>> &matrix, std::vector<std::vector<double>> &a_local);
};

}  // namespace shvetsova_k_gausse_vert_strip
