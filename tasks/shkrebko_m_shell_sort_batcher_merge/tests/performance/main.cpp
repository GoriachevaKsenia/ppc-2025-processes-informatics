#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <random>
#include <vector>

#include "shkrebko_m_shell_sort_batcher_merge/common/include/common.hpp"
#include "shkrebko_m_shell_sort_batcher_merge/mpi/include/ops_mpi.hpp"
#include "shkrebko_m_shell_sort_batcher_merge/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shkrebko_m_shell_sort_batcher_merge {

class ShkrebkoMRunPerfTestShellSortBatcherMerge : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(ShkrebkoMRunPerfTestShellSortBatcherMerge, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ShkrebkoMShellSortBatcherMergeMPI, ShkrebkoMShellSortBatcherMergeSEQ>(
        PPC_SETTINGS_shkrebko_m_shell_sort_batcher_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShkrebkoMRunPerfTestShellSortBatcherMerge::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShkrebkoMRunPerfTestShellSortBatcherMerge, kGtestValues, kPerfTestName);

}  // namespace shkrebko_m_shell_sort_batcher_merge
