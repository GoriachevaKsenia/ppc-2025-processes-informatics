#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <tuple>

#include "morozova_s_matrix_max_value/common/include/common.hpp"
#include "morozova_s_matrix_max_value/mpi/include/ops_mpi.hpp"
#include "morozova_s_matrix_max_value/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace morozova_s_matrix_max_value {

class MorozovaSRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_number = std::get<0>(params);
    switch (test_number) {
      case 1:
        input_data_ = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        break;
      case 2:
        input_data_ = {{10, 20, 30, 40}, {15, 25, 35, 45}, {12, 22, 32, 42}, {18, 28, 38, 48}};
        break;
      case 3:
        input_data_ = {{100, 200, 300, 400, 500},
                       {150, 250, 350, 450, 550},
                       {120, 220, 320, 420, 520},
                       {180, 280, 380, 480, 580},
                       {160, 260, 360, 460, 1000}};
        break;
      case 4:
        input_data_ = {};
        break;
      case 5:
        input_data_ = {{1, 2, 3}, {4, 5}};
        break;
      case 6:
        input_data_ = {{}};
        break;
      default:
        input_data_ = {{1, 2}, {3, 4}};
        break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_number = std::get<0>(params);
    if (test_number == 4 || test_number == 5 || test_number == 6) {
      return true;
    }
    int expected_max = std::numeric_limits<int>::min();
    for (const auto &row : input_data_) {
      for (int value : row) {
        expected_max = std::max(expected_max, value);
      }
    }
    return output_data == expected_max;
  }
  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {
TEST_P(MorozovaSRunFuncTestsProcesses, MatrixMaxValue) {
  ExecuteTest(GetParam());
}
const std::array<TestType, 3> kTestParamMPI = {std::make_tuple(1, "small"), std::make_tuple(2, "medium"),
                                               std::make_tuple(3, "large")};

const std::array<TestType, 6> kTestParamSEQ = {std::make_tuple(1, "small"),   std::make_tuple(2, "medium"),
                                               std::make_tuple(3, "large"),   std::make_tuple(4, "empty"),
                                               std::make_tuple(5, "invalid"), std::make_tuple(6, "zero_cols")};
const auto kTestTasksMPI =
    ppc::util::AddFuncTask<MorozovaSMatrixMaxValueMPI, InType>(kTestParamMPI, PPC_SETTINGS_morozova_s_matrix_max_value);
const auto kTestTasksSEQ =
    ppc::util::AddFuncTask<MorozovaSMatrixMaxValueSEQ, InType>(kTestParamSEQ, PPC_SETTINGS_morozova_s_matrix_max_value);
const auto kTestTasksList = std::tuple_cat(kTestTasksMPI, kTestTasksSEQ);
const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = MorozovaSRunFuncTestsProcesses::PrintFuncTestName<MorozovaSRunFuncTestsProcesses>;
INSTANTIATE_TEST_SUITE_P(MatrixMaxValueTests, MorozovaSRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace morozova_s_matrix_max_value
