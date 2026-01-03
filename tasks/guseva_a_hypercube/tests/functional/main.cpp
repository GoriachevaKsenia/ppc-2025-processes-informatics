#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "guseva_a_hypercube/common/include/common.hpp"
#include "guseva_a_hypercube/mpi/include/ops_mpi.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace guseva_a_hypercube {

class GusevaAHypercubeRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<3>(std::get<0>(test_param));
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(param);
    output_data_ = std::get<1>(param);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double precision = 10e-9;
    return output_data_ - precision <= output_data && output_data <= output_data_ + precision;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType output_data_ = 0;
};

namespace {

TEST_P(GusevaAHypercubeRunFuncTestsProcesses, GusevaAHypercubeFunc) {
  ExecuteTest(GetParam());
}

const int kRangeEnd = 1e3;

// NOLINTNEXTLINE // to avoid "Prefer 'std::number::e' to this literal" which is not possible for AppleClang 17
const double expConst = 2.7182818284590452353602874713527;

// clang-format off
const std::array<TestType, 8> kTestParam = {
  TestType(
    InType(1, kRangeEnd, [](int x) { return static_cast<double>(x); }, "x_1e3"),
    (std::pow(kRangeEnd, 2) + kRangeEnd) / 2
  ),
  TestType(
    InType(1, kRangeEnd, [](int x) { return std::pow(x, 2); }, "squared_1e3"),
    ((2 * std::pow(kRangeEnd, 3)) + (3 * std::pow(kRangeEnd, 2)) + kRangeEnd) / 6L
  ),
  TestType(
    InType(1, kRangeEnd, [](int x) { return std::log2(x); }, "log2_1e3"), 
    8529.398004204
  ),
  TestType(
    InType(1, kRangeEnd, [](int x) { return std::exp(x); }, "exp_1e3"),
    (std::exp(kRangeEnd + 1) - expConst) / (expConst - 1)
  ),
  TestType(
    InType(1, kRangeEnd, [](int x) { return std::sqrt(x); }, "sqrt_1e3"), 
    21097.4558874807
  ),
  TestType(
    InType(1, kRangeEnd, [](int x) { return std::sin(x); }, "sin_1e3"),
    (std::sin(kRangeEnd) - std::cos(kRangeEnd) / std::tan(1. / 2) + 1. / std::tan(1. / 2)) / 2
  ),
  TestType(
    InType(1, kRangeEnd, [](int x) { return std::cos(x); }, "cos_1e3"),
    (std::cos(kRangeEnd) + std::sin(kRangeEnd) / std::tan(1. / 2) - 1) / 2
  ),
  TestType(
    InType(1, kRangeEnd, [](int x) { return std::pow(x, 3); }, "cubed_1e3"),
    (std::pow(kRangeEnd * (kRangeEnd + 1), 2)) / 4
  )
};
// clang-format on

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<guseva_a_hypercube::GusevaAHypercubeMine, InType>(
    kTestParam, PPC_SETTINGS_guseva_a_hypercube));
inline const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

inline const auto kPerfTestName =
    GusevaAHypercubeRunFuncTestsProcesses::PrintFuncTestName<GusevaAHypercubeRunFuncTestsProcesses>;

// NOLINTNEXTLINE // To avoid "Variable <...> is non-cost and globally accessible, consider making it const"
INSTANTIATE_TEST_SUITE_P(GusevaAHypercubeFunc, GusevaAHypercubeRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace guseva_a_hypercube
