#include <gtest/gtest.h>

#include <cstddef>
#include <random>
#include <vector>

#include "papulina_y_gauss_filter_block/common/include/common.hpp"
#include "papulina_y_gauss_filter_block/mpi/include/ops_mpi.hpp"
#include "papulina_y_gauss_filter_block/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace papulina_y_gauss_filter_block {

class PapulinaYPerfTestsGaussFilter : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int width_ = 1200;
  const int height_ = 1200;
  const int channels_ = 3;
  InType input_data_;
  std::vector<unsigned char> expected_result_;

  void SetUp() override {
    std::mt19937 gen(height_);
    std::uniform_int_distribution<> dist(0, 255);

    std::vector<unsigned char> pixels(static_cast<size_t>(width_ * height_ * channels_));
    for (int i = 0; i < width_ * height_ * channels_; i++) {
      pixels[i] = static_cast<unsigned char>(dist(gen));
    }

    input_data_ = Picture(width_, height_, channels_, pixels);

    PapulinaYGaussFilterSEQ seq_filter(input_data_);
    seq_filter.Validation();
    seq_filter.PreProcessing();
    seq_filter.Run();
    seq_filter.PostProcessing();
    expected_result_ = seq_filter.GetOutput().pixels;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    output_data = Picture{};
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(PapulinaYPerfTestsGaussFilter, GaussFilterPerfTests) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, PapulinaYGaussFilterMPI, PapulinaYGaussFilterSEQ>(
    PPC_SETTINGS_papulina_y_gauss_filter_block);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PapulinaYPerfTestsGaussFilter::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(GaussFilterPerfTests, PapulinaYPerfTestsGaussFilter, kGtestValues, kPerfTestName);

}  // namespace papulina_y_gauss_filter_block
