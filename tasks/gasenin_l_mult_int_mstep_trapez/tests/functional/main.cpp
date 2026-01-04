#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>
#include <tuple>

#include "gasenin_l_mult_int_mstep_trapez/common/include/common.hpp"
#include "gasenin_l_mult_int_mstep_trapez/mpi/include/ops_mpi.hpp"
#include "gasenin_l_mult_int_mstep_trapez/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace gasenin_l_mult_int_mstep_trapez {

class GaseninLMultIntMstepTrapezRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &data = std::get<0>(test_param);
    std::string desc = std::get<1>(test_param);
    return desc + "_steps" + std::to_string(data.n_steps) + "_func" + std::to_string(data.func_id);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    exact_integral_ = GetExactIntegral(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if ((input_data_.func_id >= 0 && input_data_.func_id <= 3) || input_data_.func_id == 5) {
      double tolerance = (2e-2 * std::abs(exact_integral_)) + 1e-4;
      return std::abs(output_data - exact_integral_) < tolerance;
    }
    return std::isfinite(output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  double exact_integral_ = 0.0;
};

namespace {

TEST_P(GaseninLMultIntMstepTrapezRunFuncTests, IntegrationTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 17> kTestParam = {
    std::make_tuple(TaskData{.n_steps = 100, .func_id = 0, .x1 = 0.0, .x2 = 1.0, .y1 = 0.0, .y2 = 1.0},
                    "simple_unit_square_func0"),
    std::make_tuple(TaskData{.n_steps = 200, .func_id = 0, .x1 = 0.0, .x2 = 2.0, .y1 = 0.0, .y2 = 3.0},
                    "rectangle_2x3_func0"),

    std::make_tuple(TaskData{.n_steps = 150, .func_id = 1, .x1 = 0.0, .x2 = 1.0, .y1 = 0.0, .y2 = 1.0},
                    "unit_square_func1"),
    std::make_tuple(TaskData{.n_steps = 200, .func_id = 1, .x1 = 0.0, .x2 = 2.0, .y1 = 0.0, .y2 = 1.0},
                    "rectangle_2x1_func1"),

    std::make_tuple(
        TaskData{.n_steps = 120, .func_id = 2, .x1 = 0.0, .x2 = std::numbers::pi, .y1 = 0.0, .y2 = std::numbers::pi},
        "pi_square_func2"),
    std::make_tuple(TaskData{.n_steps = 100,
                             .func_id = 2,
                             .x1 = 0.0,
                             .x2 = std::numbers::pi / 2.0,
                             .y1 = 0.0,
                             .y2 = std::numbers::pi / 2.0},
                    "half_pi_square_func2"),

    std::make_tuple(TaskData{.n_steps = 100, .func_id = 3, .x1 = 0.0, .x2 = 1.0, .y1 = 0.0, .y2 = 1.0}, "exp_func3"),
    std::make_tuple(TaskData{.n_steps = 100, .func_id = 4, .x1 = 0.0, .x2 = 1.0, .y1 = 0.0, .y2 = 1.0}, "sqrt_func4"),

    std::make_tuple(TaskData{.n_steps = 50, .func_id = 5, .x1 = 0.0, .x2 = 5.0, .y1 = 0.0, .y2 = 5.0},
                    "const_func_default"),

    std::make_tuple(TaskData{.n_steps = 9, .func_id = 0, .x1 = 0.0, .x2 = 1.0, .y1 = 0.0, .y2 = 1.0},
                    "minimal_grid_func0"),

    std::make_tuple(TaskData{.n_steps = 10, .func_id = 0, .x1 = 0.0, .x2 = 1.0, .y1 = 0.0, .y2 = 1.0},
                    "small_grid_func0"),

    std::make_tuple(TaskData{.n_steps = 21, .func_id = 1, .x1 = 0.0, .x2 = 1.0, .y1 = 0.0, .y2 = 1.0},
                    "odd_steps_func1"),

    std::make_tuple(TaskData{.n_steps = 100, .func_id = 0, .x1 = 1.0, .x2 = 4.0, .y1 = 2.0, .y2 = 5.0},
                    "asymmetric_bounds_func0"),

    std::make_tuple(TaskData{.n_steps = 200, .func_id = 4, .x1 = 0.0, .x2 = 2.0, .y1 = 0.0, .y2 = 2.0},
                    "sqrt_large_grid"),

    std::make_tuple(TaskData{.n_steps = 10, .func_id = 5, .x1 = 0.0, .x2 = 2.0, .y1 = 0.0, .y2 = 3.0},
                    "const_small_grid"),

    std::make_tuple(TaskData{.n_steps = 50, .func_id = 3, .x1 = 0.0, .x2 = 0.5, .y1 = 0.0, .y2 = 0.5},
                    "exp_small_domain"),

    std::make_tuple(TaskData{.n_steps = 100,
                             .func_id = 2,
                             .x1 = 0.0,
                             .x2 = std::numbers::pi / 2.0,
                             .y1 = 0.0,
                             .y2 = std::numbers::pi / 2.0},
                    "sin_cos_small"),
};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<GaseninLMultIntMstepTrapezMPI, InType>(
                                               kTestParam, PPC_SETTINGS_gasenin_l_mult_int_mstep_trapez),
                                           ppc::util::AddFuncTask<GaseninLMultIntMstepTrapezSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_gasenin_l_mult_int_mstep_trapez));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    GaseninLMultIntMstepTrapezRunFuncTests::PrintFuncTestName<GaseninLMultIntMstepTrapezRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(IntegrationTests, GaseninLMultIntMstepTrapezRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gasenin_l_mult_int_mstep_trapez
