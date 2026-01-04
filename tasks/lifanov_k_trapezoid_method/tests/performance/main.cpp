#include <gtest/gtest.h>

#include <cmath>

#include "lifanov_k_trapezoid_method/common/include/common.hpp"
#include "lifanov_k_trapezoid_method/mpi/include/ops_mpi.hpp"
#include "lifanov_k_trapezoid_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace lifanov_k_trapezoid_method {

class LifanovKTrapezoidMethodPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType input_data;

  void SetUp() override {
    input_data = {0.0, 1.0, 0.0, 1.0, 2000.0, 2000.0};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::isfinite(output_data) && output_data > 0.0;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(LifanovKTrapezoidMethodPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, LifanovKTrapezoidMethodMPI, LifanovKTrapezoidMethodSEQ>(
    PPC_SETTINGS_lifanov_k_trapezoid_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LifanovKTrapezoidMethodPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(LifanovKTrapezoidMethodPerfTests, LifanovKTrapezoidMethodPerfTests, kGtestValues,
                         kPerfTestName);

}  // namespace lifanov_k_trapezoid_method
