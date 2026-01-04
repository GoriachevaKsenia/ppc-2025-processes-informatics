#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "liulin_y_vert_strip_diag_matrix_vect_mult/common/include/common.hpp"
#include "liulin_y_vert_strip_diag_matrix_vect_mult/mpi/include/ops_mpi.hpp"
#include "liulin_y_vert_strip_diag_matrix_vect_mult/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace liulin_y_vert_strip_diag_matrix_vect_mult {

class LiulinYVertStripDiagMatrixVectMultPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  struct TestConfig {
    int rows;
    int cols;
    std::string name;
  };

  static TestConfig GetLiulinSpecificConfig(const std::string &test_name) {
    constexpr int kLargeDim = 8192;
    if (test_name.find("liulin_y_vert_strip_diag_matrix_vect_mult_mpi") != std::string::npos) {
      return {.rows = kLargeDim, .cols = kLargeDim, .name = "mpi_large"};
    }
    if (test_name.find("liulin_y_vert_strip_diag_matrix_vect_mult_seq") != std::string::npos) {
      return {.rows = kLargeDim, .cols = kLargeDim, .name = "seq_large"};
    }
    return {.rows = 100, .cols = 100, .name = "small"};
  }

  static bool HasLiulinSpecificConfig(const std::string &test_name) {
    return test_name.find("liulin_y_vert_strip_diag_matrix_vect_mult") != std::string::npos;
  }

  static TestConfig GetTestConfig(const std::string &test_name) {
    if (HasLiulinSpecificConfig(test_name)) {
      return GetLiulinSpecificConfig(test_name);
    }
    if (test_name.find("small") != std::string::npos) {
      return {.rows = 100, .cols = 100, .name = "small"};
    }
    if (test_name.find("medium") != std::string::npos) {
      return {.rows = 1000, .cols = 1000, .name = "medium"};
    }
    if (test_name.find("large") != std::string::npos) {
      return {.rows = 5000, .cols = 5000, .name = "large"};
    }
    if (test_name.find("xlarge") != std::string::npos) {
      return {.rows = 10000, .cols = 10000, .name = "xlarge"};
    }
    if (test_name.find("tall") != std::string::npos) {
      return {.rows = 10000, .cols = 1000, .name = "tall"};
    }
    if (test_name.find("wide") != std::string::npos) {
      return {.rows = 1000, .cols = 10000, .name = "wide"};
    }
    return {.rows = 100, .cols = 100, .name = "small"};
  }

  void SetUp() override {
    const auto &test_param = GetParam();
    const std::string &test_name = std::get<1>(test_param);

    TestConfig config = GetTestConfig(test_name);
    rows_ = config.rows;
    cols_ = config.cols;

    GenerateTestData();
  }

  static int GenerateCellValue(int row, int col) {
    constexpr int kMod = 2001;
    const int raw_value = (row * 131 + col * 17) % kMod;
    return raw_value - 1000;
  }

  void GenerateTestData() {
    std::vector<std::vector<int>> matrix(rows_, std::vector<int>(cols_));
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < cols_; ++j) {
        matrix[i][j] = GenerateCellValue(i, j);
      }
    }

    std::vector<int> vect(cols_);
    for (int j = 0; j < cols_; ++j) {
      vect[j] = GenerateCellValue(0, j + 100);
    }

    input_data_ = std::make_tuple(matrix, vect);

    // Ожидаемый результат
    expected_output_.assign(rows_, 0);
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < cols_; ++j) {
        expected_output_[i] += matrix[i][j] * vect[j];
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }
    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_output_[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  int rows_ = 0;
  int cols_ = 0;
  InType input_data_;
  OutType expected_output_;
};

TEST_P(LiulinYVertStripDiagMatrixVectMultPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LiulinYVertStripDiagMatrixVectMultSEQ, LiulinYVertStripDiagMatrixVectMultMPI>(
        PPC_SETTINGS_liulin_y_vert_strip_diag_matrix_vect_mult);
const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LiulinYVertStripDiagMatrixVectMultPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LiulinYVertStripDiagMatrixVectMultPerfTests, kGtestValues, kPerfTestName);

}  // namespace liulin_y_vert_strip_diag_matrix_vect_mult
