#include <gtest/gtest.h>

#include "spichek_d_simpson_integral/common/include/common.hpp"
#include "spichek_d_simpson_integral/mpi/include/ops_mpi.hpp"
#include "spichek_d_simpson_integral/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace spichek_d_simpson_integral {

class SpichekDSimpsonIntegralRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 10000;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;  // число разбиений
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем только корректность выполнения
    return output_data >= 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SpichekDSimpsonIntegralRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SpichekDSimpsonIntegralMPI, SpichekDSimpsonIntegralSEQ>(
    PPC_SETTINGS_spichek_d_simpson_integral);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SpichekDSimpsonIntegralRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SpichekDSimpsonIntegralRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace spichek_d_simpson_integral
