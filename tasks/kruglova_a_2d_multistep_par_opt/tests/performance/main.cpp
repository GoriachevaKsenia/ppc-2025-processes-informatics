#include <gtest/gtest.h>

#include <cmath>

#include "kruglova_a_2d_multistep_par_opt/common/include/common.hpp"
#include "kruglova_a_2d_multistep_par_opt/mpi/include/ops_mpi.hpp"
#include "kruglova_a_2d_multistep_par_opt/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kruglova_a_2d_multistep_par_opt {

class KruglovaA2DMultRunPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  KruglovaA2DMultRunPerfTest() : input_data_(-5.12, 5.12, -5.12, 5.12, 1e-6, 2000) {}

 protected:
  void SetUp() override {}

  bool CheckTestOutputData(OutType &output_data) final {
    return std::abs(output_data.f_value) < 1e-1;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(KruglovaA2DMultRunPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KruglovaA2DMuitMPI, KruglovaA2DMuitSEQ>(
    PPC_SETTINGS_kruglova_a_2d_multistep_par_opt);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KruglovaA2DMultRunPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KruglovaA2DMultRunPerfTest, kGtestValues, kPerfTestName);

}  // namespace kruglova_a_2d_multistep_par_opt
