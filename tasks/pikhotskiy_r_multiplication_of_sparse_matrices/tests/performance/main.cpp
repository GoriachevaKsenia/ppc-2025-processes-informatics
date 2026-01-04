#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <vector>

#include "pikhotskiy_r_multiplication_of_sparse_matrices/common/include/common.hpp"
#include "pikhotskiy_r_multiplication_of_sparse_matrices/mpi/include/ops_mpi.hpp"
#include "pikhotskiy_r_multiplication_of_sparse_matrices/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace pikhotskiy_r_multiplication_of_sparse_matrices {

const int kMatrixSize = 300;
const double kSparsity = 0.1;

class SparseMatrixMultPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;
  SparseMatrixCRS expected_result_;

  void SetUp() override {
    std::vector<double> dense_a(static_cast<size_t>(kMatrixSize) * kMatrixSize, 0.0);
    std::vector<double> dense_b(static_cast<size_t>(kMatrixSize) * kMatrixSize, 0.0);

    int step = std::max(1, static_cast<int>(1.0 / kSparsity));

    for (int i = 0; i < kMatrixSize; ++i) {
      for (int j = 0; j < kMatrixSize; ++j) {
        if ((i + j) % step == 0) {
          dense_a[(i * kMatrixSize) + j] = static_cast<double>((i + j + 1) % 10) + 1.0;
        }
        if ((i * 2 + j) % step == 0) {
          dense_b[(i * kMatrixSize) + j] = static_cast<double>((i * j + 1) % 10) + 1.0;
        }
      }
    }

    SparseMatrixCRS mat_a = DenseToCRS(dense_a, kMatrixSize, kMatrixSize);
    SparseMatrixCRS mat_b = DenseToCRS(dense_b, kMatrixSize, kMatrixSize);

    input_data_ = std::make_tuple(mat_a, mat_b);

    std::vector<double> expected_dense(static_cast<size_t>(kMatrixSize) * kMatrixSize, 0.0);
    for (int i = 0; i < kMatrixSize; ++i) {
      for (int j = 0; j < kMatrixSize; ++j) {
        double sum = 0.0;
        for (int k = 0; k < kMatrixSize; ++k) {
          sum += dense_a[(i * kMatrixSize) + k] * dense_b[(k * kMatrixSize) + j];
        }
        expected_dense[(i * kMatrixSize) + j] = sum;
      }
    }
    expected_result_ = DenseToCRS(expected_dense, kMatrixSize, kMatrixSize);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return CompareSparseMatrices(output_data, expected_result_, 1e-9);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SparseMatrixMultPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SparseMatrixMultiplicationMPI, SparseMatrixMultiplicationSEQ>(
        PPC_SETTINGS_pikhotskiy_r_multiplication_of_sparse_matrices);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SparseMatrixMultPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SparseMatrixMultPerfTests, kGtestValues, kPerfTestName);

}  // namespace pikhotskiy_r_multiplication_of_sparse_matrices
