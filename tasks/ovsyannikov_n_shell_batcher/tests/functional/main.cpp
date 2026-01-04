#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "ovsyannikov_n_shell_batcher/common/include/common.hpp"
#include "ovsyannikov_n_shell_batcher/mpi/include/ops_mpi.hpp"
#include "ovsyannikov_n_shell_batcher/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace ovsyannikov_n_shell_batcher {

class OvsyannikovNShellBatcherFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(static_cast<uint64_t>(test_param));
  }

 protected:
  void SetUp() override {
    TestType param = std::get<2>(GetParam());

    int seed = static_cast<int>(param % 100);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(-100000, 100000);

    input_data_.resize(param);
    std::ranges::generate(input_data_, [&]() { return dist(gen); });
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::vector<int> expected = input_data_;
    std::ranges::sort(expected);
    return expected == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(OvsyannikovNShellBatcherFuncTest, ShellBatcher) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParams = {100, 1000, 10000};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<OvsyannikovNShellBatcherSEQ, InType>(kTestParams, PPC_SETTINGS_ovsyannikov_n_shell_batcher),
    ppc::util::AddFuncTask<OvsyannikovNShellBatcherMPI, InType>(kTestParams, PPC_SETTINGS_ovsyannikov_n_shell_batcher));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = OvsyannikovNShellBatcherFuncTest::PrintFuncTestName<OvsyannikovNShellBatcherFuncTest>;

INSTANTIATE_TEST_SUITE_P(ShellBatcherMergeTests, OvsyannikovNShellBatcherFuncTest, kGtestValues, kTestName);

}  // namespace ovsyannikov_n_shell_batcher
