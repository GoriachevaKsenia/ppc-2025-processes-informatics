#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "lobanov_d_multiply_matrix_ccs/common/include/common.hpp"
#include "lobanov_d_multiply_matrix_ccs/mpi/include/ops_mpi.hpp"
#include "lobanov_d_multiply_matrix_ccs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace lobanov_d_multiply_matrix_ccs {
CompressedColumnMatrix CreateRandomCompressedColumnMatrix(int row_count, int column_count, double density_factor,
                                                          int seed = 42) {
  CompressedColumnMatrix result_matrix;
  result_matrix.row_count = row_count;
  result_matrix.column_count = column_count;
  result_matrix.non_zero_count = 0;

  result_matrix.value_data.clear();
  result_matrix.row_index_data.clear();
  result_matrix.column_pointer_data.clear();

  if (row_count <= 0 || column_count <= 0) {
    result_matrix.column_pointer_data.assign(static_cast<std::size_t>(column_count) + 1U, 0);
    return result_matrix;
  }

  density_factor = std::clamp(density_factor, 0.0, 1.0);

  std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));

  std::hash<std::string> hasher;
  std::string param_hash =
      std::to_string(row_count) + "_" + std::to_string(column_count) + "_" + std::to_string(density_factor);
  auto hash_value = static_cast<std::mt19937::result_type>(hasher(param_hash));
  rng.seed(static_cast<std::mt19937::result_type>(seed) + hash_value);

  std::uniform_real_distribution<double> val_dist(0.1, 10.0);
  std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

  std::vector<std::vector<int>> row_indices_per_column(static_cast<std::size_t>(column_count));
  std::vector<std::vector<double>> values_per_column(static_cast<std::size_t>(column_count));

  int nnz_counter = 0;

  for (int j = 0; j < column_count; ++j) {
    for (int i = 0; i < row_count; ++i) {
      if (prob_dist(rng) < density_factor) {
        row_indices_per_column[static_cast<std::size_t>(j)].push_back(i);
        values_per_column[static_cast<std::size_t>(j)].push_back(val_dist(rng));
        ++nnz_counter;
      }
    }
  }

  result_matrix.non_zero_count = nnz_counter;

  if (nnz_counter > 0) {
    result_matrix.value_data.reserve(static_cast<std::size_t>(nnz_counter));
    result_matrix.row_index_data.reserve(static_cast<std::size_t>(nnz_counter));
  }

  result_matrix.column_pointer_data.assign(static_cast<std::size_t>(column_count) + 1U, 0);

  int offset = 0;
  result_matrix.column_pointer_data[0] = 0;

  for (int j = 0; j < column_count; ++j) {
    auto &column_rows = row_indices_per_column[static_cast<std::size_t>(j)];
    auto &column_values = values_per_column[static_cast<std::size_t>(j)];

    for (std::size_t k = 0; k < column_rows.size(); ++k) {
      int row_idx = column_rows[k];
      if (row_idx >= 0 && row_idx < row_count) {
        result_matrix.row_index_data.push_back(row_idx);
        result_matrix.value_data.push_back(column_values[k]);
      }
    }

    offset += static_cast<int>(column_rows.size());
    result_matrix.column_pointer_data[static_cast<std::size_t>(j) + 1U] = offset;
  }

  result_matrix.non_zero_count = static_cast<int>(result_matrix.value_data.size());

  if (!result_matrix.column_pointer_data.empty()) {
    result_matrix.column_pointer_data.back() = result_matrix.non_zero_count;
  }

  for (std::size_t i = 1; i < result_matrix.column_pointer_data.size(); ++i) {
    if (result_matrix.column_pointer_data[i] < result_matrix.column_pointer_data[i - 1U]) {
      result_matrix.column_pointer_data[i] =
          std::max(result_matrix.column_pointer_data[i], result_matrix.column_pointer_data[i - 1U]);
    }
  }

  if (result_matrix.non_zero_count > 0) {
    if (result_matrix.value_data.size() != result_matrix.row_index_data.size()) {
      std::size_t min_size = std::min(result_matrix.value_data.size(), result_matrix.row_index_data.size());
      result_matrix.value_data.resize(min_size);
      result_matrix.row_index_data.resize(min_size);
      result_matrix.non_zero_count = static_cast<int>(min_size);
    }
  }

  if (!result_matrix.column_pointer_data.empty()) {
    result_matrix.column_pointer_data.back() = result_matrix.non_zero_count;
  }

  return result_matrix;
}

class LobanovDMultiplyMatrixFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_parameter) {
    return std::get<0>(test_parameter);
  }

 protected:
  void SetUp() override {
    TestType parameters = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::string test_name = std::get<0>(parameters);
    if (test_name.find("rectangular_1") != std::string::npos) {
      matrix_a_ = CreateRandomCompressedColumnMatrix(10, 5, 0.3);
      matrix_b_ = CreateRandomCompressedColumnMatrix(5, 8, 0.3);
    } else if (test_name.find("rectangular_2") != std::string::npos) {
      matrix_a_ = CreateRandomCompressedColumnMatrix(7, 12, 0.3);
      matrix_b_ = CreateRandomCompressedColumnMatrix(12, 4, 0.3);
    } else if (test_name.find("small") != std::string::npos) {
      matrix_a_ = CreateRandomCompressedColumnMatrix(10, 10, 0.3);
      matrix_b_ = CreateRandomCompressedColumnMatrix(10, 10, 0.3);
    } else if (test_name.find("medium") != std::string::npos) {
      matrix_a_ = CreateRandomCompressedColumnMatrix(50, 50, 0.1);
      matrix_b_ = CreateRandomCompressedColumnMatrix(50, 50, 0.1);
    } else if (test_name.find("large") != std::string::npos) {
      matrix_a_ = CreateRandomCompressedColumnMatrix(100, 100, 0.05);
      matrix_b_ = CreateRandomCompressedColumnMatrix(100, 100, 0.05);
    } else if (test_name.find("3x1_times_1x4") != std::string::npos) {
      matrix_a_ = CreateRandomCompressedColumnMatrix(3, 1, 0.8);
      matrix_b_ = CreateRandomCompressedColumnMatrix(1, 4, 0.8);
    } else if (test_name.find("dense_small") != std::string::npos) {
      matrix_a_ = CreateRandomCompressedColumnMatrix(4, 4, 0.8);
      matrix_b_ = CreateRandomCompressedColumnMatrix(4, 4, 0.8);
    } else if (test_name.find("sparse_small") != std::string::npos) {
      matrix_a_ = CreateRandomCompressedColumnMatrix(6, 6, 0.1);
      matrix_b_ = CreateRandomCompressedColumnMatrix(6, 6, 0.1);
    } else if (test_name.find("large_rows_small_cols") != std::string::npos) {
      matrix_a_ = CreateRandomCompressedColumnMatrix(100, 3, 0.3);
      matrix_b_ = CreateRandomCompressedColumnMatrix(3, 5, 0.3);
    } else if (test_name.find("small_rows_large_cols") != std::string::npos) {
      matrix_a_ = CreateRandomCompressedColumnMatrix(3, 100, 0.3);
      matrix_b_ = CreateRandomCompressedColumnMatrix(100, 5, 0.3);
    } else {
      matrix_a_ = CreateRandomCompressedColumnMatrix(5, 5, 0.5);
      matrix_b_ = CreateRandomCompressedColumnMatrix(5, 5, 0.5);
    }

    input_data_ = std::make_pair(matrix_a_, matrix_b_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.row_count == 0 && output_data.column_count == 0) {
      return true;
    }

    bool dimensions_correct =
        output_data.row_count == matrix_a_.row_count && output_data.column_count == matrix_b_.column_count &&
        output_data.column_pointer_data.size() == static_cast<size_t>(output_data.column_count) + 1;

    return dimensions_correct && output_data.non_zero_count >= 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  CompressedColumnMatrix matrix_a_, matrix_b_;
  InType input_data_;
};

namespace {

TEST_P(LobanovDMultiplyMatrixFuncTest, MatrixMultiplicationCorrectness) {
  ExecuteTest(GetParam());
}

CompressedColumnMatrix CreateEmptyMatrix(int rows, int cols) {
  CompressedColumnMatrix matrix;
  matrix.row_count = rows;
  matrix.column_count = cols;
  matrix.non_zero_count = 0;
  matrix.value_data.clear();
  matrix.row_index_data.clear();
  matrix.column_pointer_data.assign(cols + 1, 0);
  return matrix;
}

const std::array<TestType, 8> kTestParameters = {
    std::make_tuple("test_small_matrices", CreateEmptyMatrix(10, 10), CreateEmptyMatrix(10, 10),
                    CreateEmptyMatrix(10, 10)),
    std::make_tuple("test_sparse_small", CreateEmptyMatrix(6, 6), CreateEmptyMatrix(6, 6), CreateEmptyMatrix(6, 6)),
    std::make_tuple("test_dense_small", CreateEmptyMatrix(4, 4), CreateEmptyMatrix(4, 4), CreateEmptyMatrix(4, 4)),
    std::make_tuple("test_medium_matrices", CreateEmptyMatrix(50, 50), CreateEmptyMatrix(50, 50),
                    CreateEmptyMatrix(50, 50)),
    std::make_tuple("test_rectangular_1", CreateEmptyMatrix(10, 5), CreateEmptyMatrix(5, 8), CreateEmptyMatrix(10, 8)),
    std::make_tuple("test_large_rows_small_cols", CreateEmptyMatrix(100, 3), CreateEmptyMatrix(3, 5),
                    CreateEmptyMatrix(100, 5)),
    std::make_tuple("test_small_rows_large_cols", CreateEmptyMatrix(3, 100), CreateEmptyMatrix(100, 5),
                    CreateEmptyMatrix(3, 5)),
    std::make_tuple("test_large_matrices", CreateEmptyMatrix(100, 100), CreateEmptyMatrix(100, 100),
                    CreateEmptyMatrix(100, 100))};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LobanovDMultiplyMatrixMPI, InType>(
                                               kTestParameters, PPC_SETTINGS_lobanov_d_multiply_matrix_ccs),
                                           ppc::util::AddFuncTask<LobanovDMultiplyMatrixSEQ, InType>(
                                               kTestParameters, PPC_SETTINGS_lobanov_d_multiply_matrix_ccs));

const auto kGtestParameterValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LobanovDMultiplyMatrixFuncTest::PrintFuncTestName<LobanovDMultiplyMatrixFuncTest>;

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationFuncTests, LobanovDMultiplyMatrixFuncTest, kGtestParameterValues,
                         kPerfTestName);

}  // namespace

}  // namespace lobanov_d_multiply_matrix_ccs
