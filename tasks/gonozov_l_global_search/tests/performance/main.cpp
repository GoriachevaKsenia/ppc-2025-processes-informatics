#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <tuple>

#include "gonozov_l_global_search/common/include/common.hpp"
#include "gonozov_l_global_search/mpi/include/ops_mpi.hpp"
#include "gonozov_l_global_search/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace gonozov_l_global_search {

class GonozovLRunGlobalSearchPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const double eps_ = 1e-7;
  std::function<double(double)> minimized_function_ = [](double x) {
    double s = 0.0;
    for (int i = 0; i < 32; ++i) {
      s += std::sin(10.0 * x) * std::sin(10.0 * x);
    }
    return (x * x) + (0.01 * s);
  };

  double a_ = -1.0;
  double b_ = 1.0;
  double r_ = 2.0;
  InType input_data_;
  OutType desired_result_ = 0.0;

  void SetUp() override {
    input_data_ = std::make_tuple(minimized_function_, r_, a_, b_, eps_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::abs(desired_result_ - output_data) <= 0.01;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(GonozovLRunGlobalSearchPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, GonozovLGlobalSearchMPI, GonozovLGlobalSearchSEQ>(
    PPC_SETTINGS_gonozov_l_global_search);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GonozovLRunGlobalSearchPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GonozovLRunGlobalSearchPerfTest, kGtestValues, kPerfTestName);

}  // namespace gonozov_l_global_search
