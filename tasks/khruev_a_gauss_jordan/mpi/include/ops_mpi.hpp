#pragma once
#include <vector>

#include "khruev_a_gauss_jordan/common/include/common.hpp"
#include "task/include/task.hpp"

namespace khruev_a_gauss_jordan {

class KhruevAGaussJordanMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit KhruevAGaussJordanMPI(const InType &in);

 private:
  std::vector<double> local_data_;
  int n_{0};
  int m_{0};
  struct PivotPos {
    double val;
    int rank;
  };
  struct RowPos {
    int rank;
    int local_idx;
  };
  static constexpr double kEps = 1e-10;
  [[nodiscard]] int GetGlobalIdx(int local_k, int rank, int size) const;

  [[nodiscard]] RowPos GetRowOwner(int global_i, int size) const;
  [[nodiscard]] int FindLocalPivotIdx(int col, int rank, int size) const;

  void SwapLocalRows(int a, int b);
  void SwapRemoteRows(int my_idx, int other_rank);
  void NormalizePivotRow(int i, const RowPos &pivot, std::vector<double> &pivot_row, int rank);
  void ApplyElimination(int i, const std::vector<double> &pivot_row, int rank, int size);

  PivotPos FindPivot(int col, int rank, int size);
  void SwapRows(int i, int pivot_rank, int rank, int size);
  void Eliminate(int i, int rank, int size);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace khruev_a_gauss_jordan
