#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "gasenin_l_image_smooth/common/include/common.hpp"
#include "gasenin_l_image_smooth/mpi/include/ops_mpi.hpp"
#include "gasenin_l_image_smooth/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace gasenin_l_image_smooth {

class GaseninLRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kWidth_ = 3840;
  const int kHeight_ = 3840;
  const int kKernelSize_ = 5;

  InType input_data_;

  void SetUp() override {
    input_data_.width = kWidth_;
    input_data_.height = kHeight_;
    input_data_.kernel_size = kKernelSize_;
    input_data_.data.resize(static_cast<size_t>(kWidth_) * static_cast<size_t>(kHeight_));

    std::ranges::fill(input_data_.data, 100);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.data.empty()) {
      return true;
    }

    return output_data.width == kWidth_ && output_data.height == kHeight_ &&
           output_data.data.size() == static_cast<size_t>(kWidth_) * static_cast<size_t>(kHeight_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(GaseninLRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, GaseninLImageSmoothMPI, GaseninLImageSmoothSEQ>(
    PPC_SETTINGS_gasenin_l_image_smooth);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GaseninLRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GaseninLRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace gasenin_l_image_smooth
