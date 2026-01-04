#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <vector>

#include "kurpiakov_a_shellsort/common/include/common.hpp"
#include "kurpiakov_a_shellsort/mpi/include/ops_mpi.hpp"
#include "kurpiakov_a_shellsort/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kurpiakov_a_shellsort {

class KurpiakovARunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kPerfSize = 100000;
  InType input_data_;
  OutType expected_data_;

  void SetUp() override {
    std::vector<int> vec(kPerfSize);
    for (int i = 0; i < kPerfSize; ++i) {
      vec[i] = kPerfSize - i;
    }

    input_data_ = std::make_tuple(kPerfSize, vec);
    std::ranges::reverse(vec);
    expected_data_ = vec;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    size_t size_output = output_data.size();
    size_t size_exp = expected_data_.size();

    if (size_output != size_exp) {
      return false;
    }

    for (size_t i = 0; i < size_exp; ++i) {
      if (output_data[i] != expected_data_[i]) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KurpiakovARunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KurpiakovAShellsortMPI, KurpiakovAShellsortSEQ>(
    PPC_SETTINGS_kurpiakov_a_shellsort);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KurpiakovARunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KurpiakovARunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace kurpiakov_a_shellsort
