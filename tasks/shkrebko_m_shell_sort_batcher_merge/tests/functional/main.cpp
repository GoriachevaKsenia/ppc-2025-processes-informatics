#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "shkrebko_m_shell_sort_batcher_merge/common/include/common.hpp"
#include "shkrebko_m_shell_sort_batcher_merge/mpi/include/ops_mpi.hpp"
#include "shkrebko_m_shell_sort_batcher_merge/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shkrebko_m_shell_sort_batcher_merge {

class ShkrebkoMRunFuncTestsShellSortBatcherMerge : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(static_cast<uint64_t>(test_param));
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int seed = static_cast<int>(param % 100);
    std::mt19937 e(seed);
    std::uniform_int_distribution<int> gen(-100000, 100000);
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

 private:
  InType input_data_;
};

namespace {

TEST_P(ShkrebkoMRunFuncTestsShellSortBatcherMerge, GetShellSortBatcherMerge) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {100, 1000, 10000};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ShkrebkoMShellSortBatcherMergeMPI, InType>(
                                               kTestParam, PPC_SETTINGS_shkrebko_m_shell_sort_batcher_merge),
                                           ppc::util::AddFuncTask<ShkrebkoMShellSortBatcherMergeSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_shkrebko_m_shell_sort_batcher_merge));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    ShkrebkoMRunFuncTestsShellSortBatcherMerge::PrintFuncTestName<ShkrebkoMRunFuncTestsShellSortBatcherMerge>;

INSTANTIATE_TEST_SUITE_P(ShellSortBatcherMergeTests, ShkrebkoMRunFuncTestsShellSortBatcherMerge, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace shkrebko_m_shell_sort_batcher_merge
