#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "ashihmin_d_calculate_integrals_by_simpson/common/include/common.hpp"
#include "ashihmin_d_calculate_integrals_by_simpson/mpi/include/ops_mpi.hpp"
#include "ashihmin_d_calculate_integrals_by_simpson/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace ashihmin_d_calculate_integrals_by_simpson {

class AshihminDCalculateIntegralsBySimpsonFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    test_input_.left_bounds = {0.0, 0.0};
    test_input_.right_bounds = {1.0, 1.0};
    test_input_.partitions = 10;
  }

  bool CheckTestOutputData(OutType &output_value) override {
    const double exact_value = 2.0 / 3.0;
    return std::abs(output_value - exact_value) < 1e-4;
  }

  InType GetTestInputData() override {
    return test_input_;
  }

 private:
  InType test_input_;
};

namespace {

TEST_P(AshihminDCalculateIntegralsBySimpsonFuncTests, Correctness) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(1, "small"), std::make_tuple(2, "medium"),
                                            std::make_tuple(3, "large")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<AshihminDCalculateIntegralsBySimpsonMPI, InType>(
                                               kTestParam, PPC_SETTINGS_ashihmin_d_calculate_integrals_by_simpson),
                                           ppc::util::AddFuncTask<AshihminDCalculateIntegralsBySimpsonSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_ashihmin_d_calculate_integrals_by_simpson));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    AshihminDCalculateIntegralsBySimpsonFuncTests::PrintFuncTestName<AshihminDCalculateIntegralsBySimpsonFuncTests>;

INSTANTIATE_TEST_SUITE_P(Correctness, AshihminDCalculateIntegralsBySimpsonFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace ashihmin_d_calculate_integrals_by_simpson
