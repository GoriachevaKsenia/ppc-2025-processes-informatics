#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "mityaeva_d_contrast_enhancement_histogram_stretching/common/include/common.hpp"
#include "mityaeva_d_contrast_enhancement_histogram_stretching/mpi/include/ops_mpi.hpp"
#include "mityaeva_d_contrast_enhancement_histogram_stretching/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace mityaeva_d_contrast_enhancement_histogram_stretching {

class ContrastEnhancementRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kImageWidth_ = 255;
  const int kImageHeight_ = 255;

 public:
  void SetUp() override {
    BaseRunPerfTests::SetUp();

    input_data_.clear();

    input_data_.push_back(static_cast<uint8_t>(kImageWidth_));
    input_data_.push_back(static_cast<uint8_t>(kImageHeight_));

    int total_pixels = kImageWidth_ * kImageHeight_;
    input_data_.reserve(2 + total_pixels);

    for (int i = 0; i < total_pixels; ++i) {
      uint8_t pixel_value = 0;

      if (i % 10 == 0) {
        pixel_value = 0;
      } else if (i % 10 == 1) {
        pixel_value = 255;
      } else if (i % 10 < 5) {
        pixel_value = static_cast<uint8_t>((i * 17 + (i / kImageWidth_) * 23) % 128);
      } else {
        pixel_value = static_cast<uint8_t>((i * 37 + (i % kImageWidth_) * 13) % 256);
      }

      input_data_.push_back(pixel_value);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() < 2) {
      return false;
    }

    if (output_data[0] != static_cast<uint8_t>(kImageWidth_)) {
      return false;
    }

    if (output_data[1] != static_cast<uint8_t>(kImageHeight_)) {
      return false;
    }

    size_t expected_size = static_cast<size_t>(kImageWidth_ * kImageHeight_) + 2;
    return output_data.size() == expected_size;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(ContrastEnhancementRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ContrastEnhancementMPI, ContrastEnhancementSEQ>(
    PPC_SETTINGS_mityaeva_d_contrast_enhancement_histogram_stretching);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ContrastEnhancementRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ContrastEnhancementRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace mityaeva_d_contrast_enhancement_histogram_stretching
