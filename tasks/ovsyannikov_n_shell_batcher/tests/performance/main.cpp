#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "ovsyannikov_n_shell_batcher/common/include/common.hpp"
#include "ovsyannikov_n_shell_batcher/mpi/include/ops_mpi.hpp"
#include "ovsyannikov_n_shell_batcher/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace ovsyannikov_n_shell_batcher {

class OvsyannikovNShellBatcherPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    TestType param = 100000;
    int seed = static_cast<int>(param % 100ULL);
    std::mt19937 e(seed);
    std::uniform_int_distribution<int> gen(-1000000, 1000000);
    std::vector<int> &vec = input_data_;
    vec.resize(param);
    std::ranges::generate(vec.begin(), vec.end(), [&]() { return gen(e); });
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::vector<int> example = input_data_;
    std::ranges::sort(example.begin(), example.end());
    return example == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(OvsyannikovNShellBatcherPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, OvsyannikovNShellBatcherMPI, OvsyannikovNShellBatcherSEQ>(
        PPC_SETTINGS_ovsyannikov_n_shell_batcher);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = OvsyannikovNShellBatcherPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, OvsyannikovNShellBatcherPerfTest, kGtestValues, kPerfTestName);
}  // namespace ovsyannikov_n_shell_batcher
