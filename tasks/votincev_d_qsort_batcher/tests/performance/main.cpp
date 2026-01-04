#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>

#include "util/include/perf_test_util.hpp"
#include "votincev_d_qsort_batcher/common/include/common.hpp"
#include "votincev_d_qsort_batcher/mpi/include/ops_mpi.hpp"
#include "votincev_d_qsort_batcher/seq/include/ops_seq.hpp"

namespace votincev_d_qsort_batcher {

class VotincevDQsortBatcherRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  InType GetTestInputData() final {
    return input_data;
  }

 protected:
  InType input_data;
  OutType expected_res;

  void SetUp() override {
    size_t vect_sz = 5000000;
    double sgn_swapper = 1;
    input_data.assign(vect_sz, 0.0);
    for (size_t i = 0; i < vect_sz; i++) {
      input_data[i] = static_cast<double>(i % 5) * sgn_swapper;
      sgn_swapper *= -1;
    }

    expected_res = input_data;
    std::ranges::sort(expected_res);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // 1,2... процессы не владеют нужным результатом
    if (output_data.size() != expected_res.size()) {
      return true;
    }
    // 0й процесс должен иметь отсортированный массив
    return output_data == expected_res;
  }
};

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, VotincevDQsortBatcherMPI, VotincevDQsortBatcherSEQ>(
    PPC_SETTINGS_votincev_d_qsort_batcher);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = VotincevDQsortBatcherRunPerfTestsProcesses::CustomPerfTestName;

TEST_P(VotincevDQsortBatcherRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(RunPerf, VotincevDQsortBatcherRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace votincev_d_qsort_batcher
