#include <gtest/gtest.h>

#include <tuple>

#include "shkrebko_m_hypercube/common/include/common.hpp"
#include "shkrebko_m_hypercube/mpi/include/ops_mpi.hpp"
#include "shkrebko_m_hypercube/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shkrebko_m_hypercube {

class ShkrebkoMHypercubePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_data_ = {100, 0};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    bool value_ok = output_data.value == input_data_[0];
    bool finish_ok = output_data.finish;

    return value_ok && finish_ok;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = {0, 0};
};

TEST_P(ShkrebkoMHypercubePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    std::tuple_cat(ppc::util::MakeAllPerfTasks<InType, ShkrebkoMHypercubeMPI>(PPC_SETTINGS_shkrebko_m_hypercube),
                   ppc::util::MakeAllPerfTasks<InType, ShkrebkoMHypercubeSEQ>(PPC_SETTINGS_shkrebko_m_hypercube));

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShkrebkoMHypercubePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShkrebkoMHypercubePerfTests, kGtestValues, kPerfTestName);

}  // namespace shkrebko_m_hypercube
