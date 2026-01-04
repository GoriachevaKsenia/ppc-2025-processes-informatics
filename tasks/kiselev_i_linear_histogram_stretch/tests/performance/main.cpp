#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "kiselev_i_linear_histogram_stretch/common/include/common.hpp"
#include "kiselev_i_linear_histogram_stretch/mpi/include/ops_mpi.hpp"
#include "kiselev_i_linear_histogram_stretch/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kiselev_i_linear_histogram_stretch {

class KiselevIRunPerfTestProcesses3 : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    constexpr std::size_t kWidth = 16000;
    constexpr std::size_t kHeight = 16000;
    const std::size_t image_size = kWidth * kHeight;

    input_data_.width = kWidth;
    input_data_.height = kHeight;
    input_data_.pixels.resize(image_size);

    for (std::size_t index = 0; index < image_size; ++index) {
      input_data_.pixels[index] = static_cast<uint8_t>(100 + (index % 51));
    }
  }

  bool CheckTestOutputData([[maybe_unused]] OutType &output_data) override {
    return true;
  }

  InType GetTestInputData() override {
    return input_data_;
  }

 private:
  InType input_data_{};
};

TEST_P(KiselevIRunPerfTestProcesses3, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KiselevITestTaskMPI, KiselevITestTaskSEQ>(
    PPC_SETTINGS_kiselev_i_linear_histogram_stretch);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KiselevIRunPerfTestProcesses3::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(LinearHistogramStretchingPerfTests, KiselevIRunPerfTestProcesses3, kGtestValues,
                         kPerfTestName);

}  // namespace kiselev_i_linear_histogram_stretch
