#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "pikhotskiy_r_multiplication_of_sparse_matrices/common/include/common.hpp"
#include "pikhotskiy_r_multiplication_of_sparse_matrices/mpi/include/ops_mpi.hpp"
#include "pikhotskiy_r_multiplication_of_sparse_matrices/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace pikhotskiy_r_multiplication_of_sparse_matrices {

class SparseMatrixMultFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::make_tuple(std::get<0>(params), std::get<1>(params));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &mat_a = std::get<0>(input_data_);
    const auto &mat_b = std::get<1>(input_data_);

    // Compute expected result using dense multiplication
    auto dense_a = CRSToDense(mat_a);
    auto dense_b = CRSToDense(mat_b);

    std::vector<double> expected_dense(static_cast<size_t>(mat_a.rows) * mat_b.cols, 0.0);
    for (int i = 0; i < mat_a.rows; ++i) {
      for (int j = 0; j < mat_b.cols; ++j) {
        double sum = 0.0;
        for (int k = 0; k < mat_a.cols; ++k) {
          sum += dense_a[(i * mat_a.cols) + k] * dense_b[(k * mat_b.cols) + j];
        }
        expected_dense[(i * mat_b.cols) + j] = sum;
      }
    }

    SparseMatrixCRS expected = DenseToCRS(expected_dense, mat_a.rows, mat_b.cols);
    return CompareSparseMatrices(output_data, expected, 1e-9);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(SparseMatrixMultFuncTests, SparseMatrixMultiplication) {
  ExecuteTest(GetParam());
}

// Helper function to create test case
TestType CreateTestCase(const std::vector<double> &dense_a, int rows_a, int cols_a, const std::vector<double> &dense_b,
                        int rows_b, int cols_b, const std::string &name) {
  SparseMatrixCRS mat_a = DenseToCRS(dense_a, rows_a, cols_a);
  SparseMatrixCRS mat_b = DenseToCRS(dense_b, rows_b, cols_b);
  return std::make_tuple(mat_a, mat_b, name);
}

// Helper to create sparse matrix with given sparsity
TestType CreateSparseTest(int rows_a, int cols_a, int cols_b, double sparsity, const std::string &name) {
  std::vector<double> dense_a(static_cast<size_t>(rows_a) * cols_a, 0.0);
  std::vector<double> dense_b(static_cast<size_t>(cols_a) * cols_b, 0.0);

  // Fill with pattern based on sparsity
  int step = std::max(1, static_cast<int>(1.0 / sparsity));

  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < cols_a; ++j) {
      if ((i + j) % step == 0) {
        dense_a[(i * cols_a) + j] = static_cast<double>((i + j + 1) % 10) + 1.0;
      }
    }
  }

  for (int i = 0; i < cols_a; ++i) {
    for (int j = 0; j < cols_b; ++j) {
      if ((i + j) % step == 0) {
        dense_b[(i * cols_b) + j] = static_cast<double>((i * j + 1) % 10) + 1.0;
      }
    }
  }

  return CreateTestCase(dense_a, rows_a, cols_a, dense_b, cols_a, cols_b, name);
}

const std::array<TestType, 12> kTestParam = {
    // Basic tests
    CreateTestCase({2.0}, 1, 1, {3.0}, 1, 1, "SingleElement"),

    CreateTestCase({1.0, 0.0, 0.0, 1.0}, 2, 2, {5.0, 6.0, 7.0, 8.0}, 2, 2, "IdentityMultiplication"),

    CreateTestCase({1.0, 2.0, 3.0, 4.0}, 2, 2, {0.0, 0.0, 0.0, 0.0}, 2, 2, "ZeroMatrixMultiplication"),

    // Sparse matrices
    CreateTestCase({1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0}, 3, 3, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
                   3, 3, "DiagonalMatrices"),

    CreateTestCase({1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0}, 3, 3, {0.0, 5.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 7.0},
                   3, 3, "SparsePattern"),

    // Rectangular matrices
    CreateTestCase({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 3, 2, "Rectangular_2x3_x_3x2"),

    CreateTestCase({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 3, 2, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 2, 3, "Rectangular_3x2_x_2x3"),

    // Row and column vectors
    CreateTestCase({1.0, 2.0, 3.0}, 1, 3, {1.0, 2.0, 3.0}, 3, 1, "RowVector_x_ColVector"),

    CreateTestCase({1.0, 2.0, 3.0}, 3, 1, {1.0, 2.0, 3.0}, 1, 3, "ColVector_x_RowVector"),

    // Larger sparse tests
    CreateSparseTest(5, 5, 5, 0.3, "Sparse_5x5_30percent"),

    CreateSparseTest(10, 10, 10, 0.2, "Sparse_10x10_20percent"),

    CreateSparseTest(8, 6, 4, 0.25, "Sparse_Rectangular_8x6x4")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<SparseMatrixMultiplicationMPI, InType>(
                       kTestParam, PPC_SETTINGS_pikhotskiy_r_multiplication_of_sparse_matrices),
                   ppc::util::AddFuncTask<SparseMatrixMultiplicationSEQ, InType>(
                       kTestParam, PPC_SETTINGS_pikhotskiy_r_multiplication_of_sparse_matrices));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SparseMatrixMultFuncTests::PrintFuncTestName<SparseMatrixMultFuncTests>;

INSTANTIATE_TEST_SUITE_P(SparseMatrixMultTests, SparseMatrixMultFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace pikhotskiy_r_multiplication_of_sparse_matrices
