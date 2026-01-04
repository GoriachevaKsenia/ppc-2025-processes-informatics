#include <gtest/gtest.h>

#include <cstddef>

#include "egorova_l_a_broadcast/common/include/common.hpp"
#include "egorova_l_a_broadcast/mpi/include/ops_mpi.hpp"
#include "egorova_l_a_broadcast/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace egorova_l_a_broadcast {

class EgorovaLRunPerfTestProcesses2 : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 45000000;
  InType input_data_{};

  void SetUp() override {
    input_data_.root = 0;
    input_data_.type_indicator = 0;
    input_data_.data_int.assign(kCount_, 42);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.size() == static_cast<size_t>(kCount_) * sizeof(int);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(EgorovaLRunPerfTestProcesses2, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, EgorovaLBroadcastMPI, EgorovaLBroadcastSEQ>(PPC_SETTINGS_egorova_l_a_broadcast);
INSTANTIATE_TEST_SUITE_P(RunModeTests, EgorovaLRunPerfTestProcesses2, ppc::util::TupleToGTestValues(kAllPerfTasks),
                         EgorovaLRunPerfTestProcesses2::CustomPerfTestName);

}  // namespace egorova_l_a_broadcast
