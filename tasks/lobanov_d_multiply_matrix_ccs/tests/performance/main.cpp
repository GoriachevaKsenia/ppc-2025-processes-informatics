#include <gtest/gtest.h>

#include <algorithm>
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
#include "util/include/perf_test_util.hpp"

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

class LobanovDMultiplyMatrixPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    auto params = GetTestParams();
    int dimension = std::get<0>(params);
    double density = std::get<1>(params);
    int seed_offset = std::get<2>(params);

    first_matrix_ = CreateRandomCompressedColumnMatrix(dimension, dimension, density, 100 + seed_offset);
    second_matrix_ = CreateRandomCompressedColumnMatrix(dimension, dimension, density, 200 + seed_offset);

    input_data_ = std::make_pair(first_matrix_, second_matrix_);
  }

  [[nodiscard]] virtual std::tuple<int, double, int> GetTestParams() const {
    return {2000, 0.1, 0};  // По умолчанию
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.row_count == 0 && output_data.column_count == 0) {
      return true;
    }

    return output_data.row_count == first_matrix_.row_count &&
           output_data.column_count == second_matrix_.column_count &&
           output_data.column_pointer_data.size() == static_cast<size_t>(output_data.column_count) + 1;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  CompressedColumnMatrix first_matrix_, second_matrix_;
  InType input_data_;
};

// Тест 1: Малые матрицы (быстрый запуск)
class SmallMatrixPerfTest : public LobanovDMultiplyMatrixPerfTest {
 protected:
  [[nodiscard]] std::tuple<int, double, int> GetTestParams() const override {
    return {500, 0.1, 1};  // 500x500, плотность 10%
  }
};

// Тест 2: Средние матрицы
class MediumMatrixPerfTest : public LobanovDMultiplyMatrixPerfTest {
 protected:
  [[nodiscard]] std::tuple<int, double, int> GetTestParams() const override {
    return {1000, 0.08, 2};  // 1000x1000, плотность 8%
  }
};

// Тест 3: Большие матрицы
class LargeMatrixPerfTest : public LobanovDMultiplyMatrixPerfTest {
 protected:
  [[nodiscard]] std::tuple<int, double, int> GetTestParams() const override {
    return {3000, 0.05, 3};  // 3000x3000, плотность 5%
  }
};

// Тесты для разных размеров матриц
TEST_P(SmallMatrixPerfTest, SmallMatrixPerformance) {
  ExecuteTest(GetParam());
}

TEST_P(MediumMatrixPerfTest, MediumMatrixPerformance) {
  ExecuteTest(GetParam());
}

TEST_P(LargeMatrixPerfTest, LargeMatrixPerformance) {
  ExecuteTest(GetParam());
}

const auto kAllPerformanceTasks =
    ppc::util::MakeAllPerfTasks<InType, LobanovDMultiplyMatrixMPI, LobanovDMultiplyMatrixSEQ>(
        PPC_SETTINGS_lobanov_d_multiply_matrix_ccs);

const auto kGtestPerformanceValues = ppc::util::TupleToGTestValues(kAllPerformanceTasks);

INSTANTIATE_TEST_SUITE_P(SmallMatrixTests, SmallMatrixPerfTest, kGtestPerformanceValues,
                         LobanovDMultiplyMatrixPerfTest::CustomPerfTestName);

INSTANTIATE_TEST_SUITE_P(MediumMatrixTests, MediumMatrixPerfTest, kGtestPerformanceValues,
                         LobanovDMultiplyMatrixPerfTest::CustomPerfTestName);

INSTANTIATE_TEST_SUITE_P(LargeMatrixTests, LargeMatrixPerfTest, kGtestPerformanceValues,
                         LobanovDMultiplyMatrixPerfTest::CustomPerfTestName);

}  // namespace lobanov_d_multiply_matrix_ccs
