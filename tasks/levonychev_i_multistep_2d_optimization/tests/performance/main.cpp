#include <gtest/gtest.h>

#include <cmath>

#include "levonychev_i_multistep_2d_optimization/common/include/common.hpp"
#include "levonychev_i_multistep_2d_optimization/mpi/include/ops_mpi.hpp"
#include "levonychev_i_multistep_2d_optimization/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace levonychev_i_multistep_2d_optimization {

static inline double RosenbrockFunction(double x, double y) {
  double term1 = 1.0 - x;
  double term2 = y - (x * x);
  return (term1 * term1) + (100.0 * term2 * term2);
}

class LevonychevIMultistep2dOptimizationPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};
  double expected_x_min_ = 0.0;
  double expected_y_min_ = 0.0;
  double expected_value_min_ = 0.0;

  void SetUp() override {
    auto func = [](double x, double y) -> double { return RosenbrockFunction(x, y); };

    input_data_.func = func;
    input_data_.x_min = -2.0;
    input_data_.x_max = 2.0;
    input_data_.y_min = -2.0;
    input_data_.y_max = 2.0;
    input_data_.num_steps = 6;
    input_data_.grid_size_step1 = 100;
    input_data_.candidates_per_step = 10;
    input_data_.use_local_optimization = true;

    expected_x_min_ = 1.0;
    expected_y_min_ = 1.0;
    expected_value_min_ = RosenbrockFunction(1.0, 1.0);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.x_min < input_data_.x_min || output_data.x_min > input_data_.x_max ||
        output_data.y_min < input_data_.y_min || output_data.y_min > input_data_.y_max) {
      return false;
    }

    if (!std::isfinite(output_data.value)) {
      return false;
    }
    const double tolerance_coord = 0.01;
    const double tolerance_value = 0.01;
    bool x_ok = std::abs(output_data.x_min - expected_x_min_) < tolerance_coord;
    bool y_ok = std::abs(output_data.y_min - expected_y_min_) < tolerance_coord;
    bool value_ok = std::abs(output_data.value - expected_value_min_) < tolerance_value;

    bool value_reasonable = output_data.value < expected_value_min_ + 20.0;

    return x_ok && y_ok && value_ok && value_reasonable;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LevonychevIMultistep2dOptimizationPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LevonychevIMultistep2dOptimizationMPI, LevonychevIMultistep2dOptimizationSEQ>(
        PPC_SETTINGS_levonychev_i_multistep_2d_optimization);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LevonychevIMultistep2dOptimizationPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LevonychevIMultistep2dOptimizationPerfTests, kGtestValues, kPerfTestName);

}  // namespace levonychev_i_multistep_2d_optimization
