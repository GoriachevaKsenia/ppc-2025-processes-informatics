#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "dolov_v_monte_carlo_integration/common/include/common.hpp"
#include "dolov_v_monte_carlo_integration/mpi/include/ops_mpi.hpp"
#include "dolov_v_monte_carlo_integration/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace dolov_v_monte_carlo_integration {
namespace {

using InType = InputParams;
using OutType = double;
using TestType = std::tuple<int, std::string>;

double FuncSquareSum(const std::vector<double> &x) {
  double sum_val = 0.0;
  for (double val : x) {
    sum_val += val * val;
  }
  return sum_val;
}

class MonteCarloHyperCubeTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int samples = std::get<0>(params);

    input_data_ = {.func = FuncSquareSum,
                   .dimension = 2,
                   .samples_count = samples,
                   .center = {0.0, 0.0},
                   .radius = 1.0,
                   .domain_type = IntegrationDomain::kHyperCube};
    expected_result_ = 8.0 / 3.0;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::abs(output_data - expected_result_) < 0.1;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_result_ = 0.0;
};

class MonteCarloHyperSphereTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int samples = std::get<0>(params);

    input_data_ = {.func = FuncSquareSum,
                   .dimension = 2,
                   .samples_count = samples,
                   .center = {0.0, 0.0},
                   .radius = 1.0,
                   .domain_type = IntegrationDomain::kHyperSphere};
    expected_result_ = std::acos(-1.0) / 2.0;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::abs(output_data - expected_result_) < 0.1;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_result_ = 0.0;
};

TEST_P(MonteCarloHyperCubeTests, IntegrationHyperCube2D) {
  ExecuteTest(GetParam());
}
TEST_P(MonteCarloHyperSphereTests, IntegrationHyperSphere2D) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParamsArray = {std::make_tuple(10000, "small"), std::make_tuple(50000, "medium"),
                                                  std::make_tuple(200000, "large")};

const auto kCubeTasksList = std::tuple_cat(ppc::util::AddFuncTask<DolovVMonteCarloIntegrationMPI, InType>(
                                               kTestParamsArray, PPC_SETTINGS_dolov_v_monte_carlo_integration),
                                           ppc::util::AddFuncTask<DolovVMonteCarloIntegrationSEQ, InType>(
                                               kTestParamsArray, PPC_SETTINGS_dolov_v_monte_carlo_integration));

const auto kCubeValsList = ppc::util::ExpandToValues(kCubeTasksList);
const auto kCubeNameStr = MonteCarloHyperCubeTests::PrintFuncTestName<MonteCarloHyperCubeTests>;
INSTANTIATE_TEST_SUITE_P(MonteCarloHyperCubeTests, MonteCarloHyperCubeTests, kCubeValsList, kCubeNameStr);

const auto kSphereTasksList = std::tuple_cat(ppc::util::AddFuncTask<DolovVMonteCarloIntegrationMPI, InType>(
                                                 kTestParamsArray, PPC_SETTINGS_dolov_v_monte_carlo_integration),
                                             ppc::util::AddFuncTask<DolovVMonteCarloIntegrationSEQ, InType>(
                                                 kTestParamsArray, PPC_SETTINGS_dolov_v_monte_carlo_integration));

const auto kSphereValsList = ppc::util::ExpandToValues(kSphereTasksList);
const auto kSphereNameStr = MonteCarloHyperSphereTests::PrintFuncTestName<MonteCarloHyperSphereTests>;
INSTANTIATE_TEST_SUITE_P(MonteCarloHyperSphereTests, MonteCarloHyperSphereTests, kSphereValsList, kSphereNameStr);

}  // namespace
}  // namespace dolov_v_monte_carlo_integration
