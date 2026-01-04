#include <gtest/gtest.h>

#include <cmath>

#include "ashihmin_d_calculate_integrals_by_simpson/common/include/common.hpp"
#include "ashihmin_d_calculate_integrals_by_simpson/mpi/include/ops_mpi.hpp"
#include "ashihmin_d_calculate_integrals_by_simpson/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace ashihmin_d_calculate_integrals_by_simpson {

class SimpsonPerformanceTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    test_input_.left_bounds = {0.0, 0.0, 0.0};
    test_input_.right_bounds = {1.0, 1.0, 1.0};
    test_input_.partitions = 20;
  }

  bool CheckTestOutputData(OutType &output_value) final {
    const double exact_value = 1.0;
    return std::abs(output_value - exact_value) < 1e-4;
  }

  InType GetTestInputData() final {
    return test_input_;
  }

 private:
  InType test_input_;
};

TEST_P(SimpsonPerformanceTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kPerformanceTasks = ppc::util::MakeAllPerfTasks<InType, AshihminDCalculateIntegralsBySimpsonMPI,
                                                           AshihminDCalculateIntegralsBySimpsonSEQ>(
    PPC_SETTINGS_ashihmin_d_calculate_integrals_by_simpson);

const auto kGtestValues = ppc::util::TupleToGTestValues(kPerformanceTasks);

INSTANTIATE_TEST_SUITE_P(PerformanceTests, SimpsonPerformanceTests, kGtestValues,
                         SimpsonPerformanceTests::CustomPerfTestName);

}  // namespace ashihmin_d_calculate_integrals_by_simpson
