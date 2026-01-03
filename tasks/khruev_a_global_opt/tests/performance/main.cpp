#include <gtest/gtest.h>

#include "khruev_a_global_opt/common/include/common.hpp"
#include "khruev_a_global_opt/mpi/include/ops_mpi.hpp"
#include "khruev_a_global_opt/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace khruev_a_global_opt {

class KhruevAGlobalOptPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType input_data{};

  void SetUp() override {
    input_data.func_id = 2;
    input_data.ax = -3.0;
    input_data.bx = 3.0;
    input_data.ay = -3.0;
    input_data.by = 3.0;

    input_data.epsilon = 1e-7;
    input_data.max_iter = 6000;
    input_data.r = 3;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.value < 1e10;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

namespace {

TEST_P(KhruevAGlobalOptPerfTests, RunPerformance) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KhruevAGlobalOptMPI, KhruevAGlobalOptSEQ>(PPC_SETTINGS_khruev_a_global_opt);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KhruevAGlobalOptPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KhruevAGlobalOptPerfTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace khruev_a_global_opt
