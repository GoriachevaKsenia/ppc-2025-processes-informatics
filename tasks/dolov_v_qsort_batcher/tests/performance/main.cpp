#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>

#include "dolov_v_qsort_batcher/common/include/common.hpp"
#include "dolov_v_qsort_batcher/mpi/include/ops_mpi.hpp"
#include "dolov_v_qsort_batcher/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dolov_v_qsort_batcher {

class DolovVQsortBatcherPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType input_data;

  void SetUp() override {
    const int k_count = 1000000;
    input_data.resize(k_count);
    for (int i = 0; i < k_count; ++i) {
      uint64_t val = ((static_cast<uint64_t>(i) * 1103515245ULL) + 12345ULL);
      input_data[i] = static_cast<double>(val % 2147483647ULL) / 1000.0;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return true;
    }
    return std::is_sorted(output_data.data(), output_data.data() + output_data.size());
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

namespace {

TEST_P(DolovVQsortBatcherPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, DolovVQsortBatcherMPI, DolovVQsortBatcherSEQ>(
    PPC_SETTINGS_dolov_v_qsort_batcher);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = DolovVQsortBatcherPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, DolovVQsortBatcherPerfTests, kGtestValues, kPerfTestName);

}  // namespace
}  // namespace dolov_v_qsort_batcher
