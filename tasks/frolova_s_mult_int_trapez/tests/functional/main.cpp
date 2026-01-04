#include <gtest/gtest.h>
// #include <mpi.h>

// #include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
// #include <cstdint>
#include <random>
#include <string>
#include <tuple>
// #include <utility>
#include <vector>

#include "frolova_s_mult_int_trapez/common/include/common.hpp"
#include "frolova_s_mult_int_trapez/mpi/include/ops_mpi.hpp"
#include "frolova_s_mult_int_trapez/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace frolova_s_mult_int_trapez {

class FrolovaRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    unsigned int test_id = std::get<0>(params);
    // std::string test_type = std::get<1>(params);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 5.0);

    input_data_.limits.clear();
    input_data_.number_of_intervals.clear();

    if (test_id == 1 || test_id == 4) {
      // 2D тест
      input_data_.limits = {{0.0, 2.0}, {0.0, 3.0}};
      input_data_.number_of_intervals = {50, 50};
      input_data_.function = [](std::vector<double> input) { return std::pow(input[0], 2) + std::pow(input[1], 2); };
    } else if (test_id == 2 || test_id == 5) {
      // 3D тест
      input_data_.limits = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
      input_data_.number_of_intervals = {30, 30, 30};
      input_data_.function = [](std::vector<double> input) { return input[0] * input[1] * input[2]; };
    } else {
      // 1D тест
      input_data_.limits = {{0.0, 3.14}};
      input_data_.number_of_intervals = {100};
      input_data_.function = [](std::vector<double> input) { return std::sin(input[0]); };
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::isfinite(output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(FrolovaRunFuncTestsProcesses, TrapezoidalIntegration) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {std::make_tuple(1U, "quadratic"), std::make_tuple(2U, "cubic"),
                                            std::make_tuple(3U, "sine"),      std::make_tuple(4U, "quadratic2"),
                                            std::make_tuple(5U, "cubic2"),    std::make_tuple(6U, "sine2")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<FrolovaSMultIntTrapezMPI, InType>(kTestParam, PPC_SETTINGS_frolova_s_mult_int_trapez),
    ppc::util::AddFuncTask<FrolovaSMultIntTrapezSEQ, InType>(kTestParam, PPC_SETTINGS_frolova_s_mult_int_trapez));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = FrolovaRunFuncTestsProcesses::PrintFuncTestName<FrolovaRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(TrapezoidalIntegrationTests, FrolovaRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace frolova_s_mult_int_trapez
