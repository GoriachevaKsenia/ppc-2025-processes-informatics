#include <gtest/gtest.h>

#include <vector>

#include "sakharov_a_transmission_from_one_to_all/common/include/common.hpp"
#include "sakharov_a_transmission_from_one_to_all/mpi/include/ops_mpi.hpp"
#include "sakharov_a_transmission_from_one_to_all/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sakharov_a_transmission_from_one_to_all {

class SakharovATransmissionPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kCount = 10000000;
  InType input_data_;
  OutType expected_result_;

  void SetUp() override {
    std::vector<int> data(kCount, 1);
    input_data_ = InType{0, data};
    expected_result_ = data;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_result_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SakharovATransmissionPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SakharovATransmissionFromOneToAllMPI, SakharovATransmissionFromOneToAllSEQ>(
        PPC_SETTINGS_sakharov_a_transmission_from_one_to_all);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SakharovATransmissionPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SakharovATransmissionPerfTests, kGtestValues, kPerfTestName);

}  // namespace sakharov_a_transmission_from_one_to_all
