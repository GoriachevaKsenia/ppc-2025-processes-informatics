#include <gtest/gtest.h>

#include "gusev_d_star/common/include/common.hpp"
#include "gusev_d_star/mpi/include/ops_mpi.hpp"
#include "gusev_d_star/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace gusev_d_star {

class GusevDStarPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    input_data_ = 500;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

TEST_P(GusevDStarPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, GusevDStarMPI, GusevDStarSEQ>(PPC_SETTINGS_gusev_d_star);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GusevDStarPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GusevDStarPerfTests, kGtestValues, kPerfTestName);

}  // namespace gusev_d_star
