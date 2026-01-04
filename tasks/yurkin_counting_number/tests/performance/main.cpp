#include <gtest/gtest.h>

#include <cctype>
#include <cstddef>

#include "util/include/perf_test_util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kCount = 100000;
  InType input_data_;

  void SetUp() override {
    input_data_ = InType(static_cast<std::size_t>(kCount), 'a');
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int expected = 0;
    for (char c : input_data_) {
      if (std::isalpha(static_cast<unsigned char>(c)) != 0) {
        ++expected;
      }
    }
    return expected == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(YurkinCountingNumberPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, YurkinCountingNumberMPI, YurkinCountingNumberSEQ>(
    PPC_SETTINGS_yurkin_counting_number);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = YurkinCountingNumberPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, YurkinCountingNumberPerfTests, kGtestValues, kPerfTestName);

}  // namespace yurkin_counting_number
