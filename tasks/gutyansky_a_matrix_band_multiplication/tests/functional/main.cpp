#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gutyansky_a_matrix_band_multiplication/common/include/common.hpp"
#include "gutyansky_a_matrix_band_multiplication/common/include/matrix.hpp"
#include "gutyansky_a_matrix_band_multiplication/mpi/include/ops_mpi.hpp"
#include "gutyansky_a_matrix_band_multiplication/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace gutyansky_a_matrix_band_multiplication {

class GutyanskyAMatrixBandMultiplicationFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    if (ShouldLoadDataAndTest()) {
      LoadTestDataFromFile();
      return;
    }

    InitializeEmptyData();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ShouldLoadDataAndTest()) {
      return output_data_ == output_data;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType output_data_ = {};

  [[nodiscard]] static bool ShouldLoadDataAndTest() {
    const std::string &test_name =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(GetParam());

    if (test_name.find("_mpi") == std::string::npos) {
      return true;
    }

    return !ppc::util::IsUnderMpirun() || ppc::util::GetMPIRank() == 0;
  }

  void InitializeEmptyData() {
    input_data_ = std::make_pair(Matrix{.rows = 0, .cols = 0, .data = {}}, Matrix{.rows = 0, .cols = 0, .data = {}});

    output_data_ = {.rows = 0, .cols = 0, .data = {}};
  }

  void LoadTestDataFromFile() {
    std::string file_name =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam()) + ".txt";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_gutyansky_a_matrix_band_multiplication, file_name);

    std::ifstream ifs(abs_path);

    if (!ifs.is_open()) {
      throw std::runtime_error("Failed to open test file: " + file_name);
    }

    size_t rows_a = 0;
    size_t cols_a = 0;
    size_t cols_b = 0;
    ifs >> rows_a >> cols_a >> cols_b;

    if (rows_a == 0 || cols_a == 0 || cols_b == 0) {
      throw std::runtime_error("All dimensions of matrices must be positive integers");
    }

    std::vector<int32_t> input_elements_a(rows_a * cols_a);
    for (size_t i = 0; i < input_elements_a.size(); i++) {
      ifs >> input_elements_a[i];
    }

    std::vector<int32_t> input_elements_b(cols_a * cols_b);
    for (size_t i = 0; i < input_elements_b.size(); i++) {
      ifs >> input_elements_b[i];
    }

    std::vector<int32_t> output_elements(rows_a * cols_b);
    for (size_t i = 0; i < output_elements.size(); i++) {
      ifs >> output_elements[i];
    }

    input_data_ = std::make_pair(Matrix{.rows = rows_a, .cols = cols_a, .data = input_elements_a},
                                 Matrix{.rows = cols_a, .cols = cols_b, .data = input_elements_b});
    output_data_ = {.rows = rows_a, .cols = cols_b, .data = output_elements};
  }
};

namespace {

TEST_P(GutyanskyAMatrixBandMultiplicationFuncTests, MatrixBandMultiplication) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 9> kTestParam = {"test_1", "test_2", "test_3", "test_4", "test_5",
                                            "test_6", "test_7", "test_8", "test_9"};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<GutyanskyAMatrixBandMultiplicationMPI, InType>(
                                               kTestParam, PPC_SETTINGS_gutyansky_a_matrix_band_multiplication),
                                           ppc::util::AddFuncTask<GutyanskyAMatrixBandMultiplicationSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_gutyansky_a_matrix_band_multiplication));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    GutyanskyAMatrixBandMultiplicationFuncTests::PrintFuncTestName<GutyanskyAMatrixBandMultiplicationFuncTests>;

INSTANTIATE_TEST_SUITE_P(ColumnSumTests, GutyanskyAMatrixBandMultiplicationFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gutyansky_a_matrix_band_multiplication
