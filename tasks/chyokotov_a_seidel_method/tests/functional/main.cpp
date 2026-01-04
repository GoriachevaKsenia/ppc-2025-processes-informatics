#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "chyokotov_a_seidel_method/common/include/common.hpp"
#include "chyokotov_a_seidel_method/mpi/include/ops_mpi.hpp"
#include "chyokotov_a_seidel_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace chyokotov_a_seidel_method {

class ChyokotovASeidelMethodFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &matrix = std::get<0>(test_param).first;
    if (matrix.empty()) {
      return "empty_matrix";
    }
    return "size_of_matrix_" + std::to_string(matrix.size()) + "x" + std::to_string(matrix[0].size());
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_output_[i]) > 0.001) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(ChyokotovASeidelMethodFuncTest, GaussSeidelMethod) {
  ExecuteTest(GetParam());
}

const std::vector<std::vector<double>> kEmptyMatrix = {};
const std::vector<double> kExpectedEmpty = {};

const std::vector<std::vector<double>> kOneElement = {{1.0}};
const std::vector<double> kExpectedOneElement = {1.0};

const std::vector<std::vector<double>> kMatrix2 = {{7.0, 2.0}, {1.0, 6.0}};
const std::vector<double> kVector2 = {38.0, 34.0};
const std::vector<double> kExpectedMatrix2 = {4.0, 5.0};

const std::vector<std::vector<double>> kMatrixWithNegativeNumber = {
    {-7.0, 1.0, 2.0}, {1.0, -8.0, 1.0}, {2.0, 1.0, -9.0}};
const std::vector<double> kVectorWithNegariveNumber = {-4.0, -6.0, -6.0};
const std::vector<double> kExpectedWithNegativeNumber = {1.0, 1.0, 1.0};

const std::vector<std::vector<double>> kMatrixWithFractionalNumber = {
    {9.3333333333, 0.3333333333, 0.6666666667, 0.1428571429},
    {0.3333333333, 10.6666666667, 0.1428571429, 0.6666666667},
    {0.6666666667, 0.1428571429, 11.1428571429, 0.3333333333},
    {0.1428571429, 0.6666666667, 0.3333333333, 12.3333333333}};
const std::vector<double> kVectorWithFractionalNumber = {3.4901960, 7.5263295, 2.052369, 5.825396825};
const std::vector<double> kExpectedMatrixWithFractionalNumber = {1.0 / 3.0, 2.0 / 3.0, 1.0 / 7.0, 3.0 / 7.0};

const std::array<TestType, 5> kTestParam = {
    std::make_tuple(std::make_pair(kEmptyMatrix, kExpectedEmpty), kExpectedEmpty),
    std::make_tuple(std::make_pair(kOneElement, kExpectedOneElement), kExpectedOneElement),
    std::make_tuple(std::make_pair(kMatrix2, kVector2), kExpectedMatrix2),
    std::make_tuple(std::make_pair(kMatrixWithNegativeNumber, kVectorWithNegariveNumber), kExpectedWithNegativeNumber),
    std::make_tuple(std::make_pair(kMatrixWithFractionalNumber, kVectorWithFractionalNumber),
                    kExpectedMatrixWithFractionalNumber)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ChyokotovASeidelMethodMPI, InType>(kTestParam, PPC_SETTINGS_chyokotov_a_seidel_method),
    ppc::util::AddFuncTask<ChyokotovASeidelMethodSEQ, InType>(kTestParam, PPC_SETTINGS_chyokotov_a_seidel_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ChyokotovASeidelMethodFuncTest::PrintFuncTestName<ChyokotovASeidelMethodFuncTest>;

INSTANTIATE_TEST_SUITE_P(GaussSeidelMethodTests, ChyokotovASeidelMethodFuncTest, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace chyokotov_a_seidel_method
