#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "zenin_a_gauss_filter/common/include/common.hpp"
#include "zenin_a_gauss_filter/mpi/include/ops_mpi.hpp"
#include "zenin_a_gauss_filter/seq/include/ops_seq.hpp"

namespace zenin_a_gauss_filter {

class ZeninAGaussFilterPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  const int width_ = 1500;
  const int height_ = 1500;
  const int channels_ = 3;

  InType input_data_;

  void SetUp() override {
    std::mt19937 gen(static_cast<std::mt19937::result_type>((height_ * 1337U) + width_));
    std::uniform_int_distribution<int> dist(0, 255);

    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width_) * height_ * channels_);
    for (auto &v : pixels) {
      v = static_cast<std::uint8_t>(dist(gen));
    }

    input_data_ = Image(height_, width_, channels_, std::move(pixels));
  }
  bool CheckTestOutputData(OutType &output_data) final {
    return (input_data_.width == output_data.width && input_data_.height == output_data.height &&
            input_data_.channels == output_data.channels && !output_data.pixels.empty());
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ZeninAGaussFilterPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ZeninAGaussFilterMPI, ZeninAGaussFilterSEQ>(PPC_SETTINGS_zenin_a_gauss_filter);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZeninAGaussFilterPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(ZeninAPerfTestGaussFilter, ZeninAGaussFilterPerfTests, kGtestValues, kPerfTestName);

}  // namespace zenin_a_gauss_filter
