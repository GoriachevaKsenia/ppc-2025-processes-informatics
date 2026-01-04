#include <gtest/gtest.h>

#include <algorithm>

#include "chaschin_v_broadcast/common/include/common.hpp"
#include "chaschin_v_broadcast/mpi/include/ops_mpi.hpp"
#include "chaschin_v_broadcast/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace chaschin_v_broadcast {

class ChaschinVRunPerfTestProcessesBR : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 10000000;
  InType input_data_;

  void SetUp() override {
    input_data_.resize(kCount_, 7863453);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::equal(input_data_.begin(), input_data_.end(), output_data.begin());
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ChaschinVRunPerfTestProcessesBR, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ChaschinVBroadcastMPI<int>, ChaschinVBroadcastSEQ<int>>(
    PPC_SETTINGS_chaschin_v_broadcast);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ChaschinVRunPerfTestProcessesBR::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ChaschinVRunPerfTestProcessesBR, kGtestValues, kPerfTestName);

}  // namespace chaschin_v_broadcast
