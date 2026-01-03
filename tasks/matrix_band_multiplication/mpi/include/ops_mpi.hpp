#pragma once

#include <cstddef>
#include <vector>

#include "matrix_band_multiplication/common/include/common.hpp"
#include "task/include/task.hpp"

namespace matrix_band_multiplication {

class MatrixBandMultiplicationMpi : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit MatrixBandMultiplicationMpi(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  void MultiplyStripe(const double *stripe_data, int stripe_cols, int stripe_offset, int local_rows);
  bool BroadcastDimensions(const Matrix &matrix_a, const Matrix &matrix_b);
  void PrepareRowDistribution(const Matrix &matrix_a);
  void PrepareColumnDistribution(const Matrix &matrix_b);
  void PrepareResultGatherInfo();
  [[nodiscard]] int ComputeMaxColumns() const;
  void PreparePackedColumns(const Matrix &matrix_b, std::vector<double> &packed, std::vector<int> &send_counts,
                            std::vector<int> &send_displs) const;
  void ScatterInitialStripe(const std::vector<double> &packed, const std::vector<int> &send_counts,
                            const std::vector<int> &send_displs, int recv_elements);

  int rank_ = 0;
  int world_size_ = 1;
  std::size_t rows_a_ = 0;
  std::size_t cols_a_ = 0;
  std::size_t rows_b_ = 0;
  std::size_t cols_b_ = 0;
  std::vector<double> local_a_;
  std::vector<double> current_b_;
  std::vector<double> rotation_buffer_;
  std::vector<double> local_result_;
  std::vector<int> row_counts_;
  std::vector<int> row_displs_;
  std::vector<int> result_counts_;
  std::vector<int> result_displs_;
  std::vector<int> col_counts_;
  std::vector<int> col_displs_;
  int stripe_owner_ = 0;
  int current_cols_ = 0;
  int max_cols_per_proc_ = 0;
};

}  // namespace matrix_band_multiplication
