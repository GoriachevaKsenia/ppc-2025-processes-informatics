#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <utility>

#include "gutyansky_a_matrix_band_multiplication/common/include/common.hpp"
#include "gutyansky_a_matrix_band_multiplication/common/include/matrix.hpp"
#include "gutyansky_a_matrix_band_multiplication/mpi/include/ops_mpi.hpp"
#include "gutyansky_a_matrix_band_multiplication/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace gutyansky_a_matrix_band_multiplication {

class GutyanskyAMatrixBandMultiplicationPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    if (ShouldLoadDataAndTest()) {
      LoadTestData();
      return;
    }

    InitializeEmptyData();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ShouldLoadDataAndTest()) {
      return output_data_ == output_data;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  const size_t kSize_ = 1000;
  InType input_data_;
  OutType output_data_ = {};

  [[nodiscard]] static bool ShouldLoadDataAndTest() {
    const std::string &test_name =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(GetParam());

    if (test_name.find("_mpi") == std::string::npos) {
      return true;
    }

    return !ppc::util::IsUnderMpirun() || ppc::util::GetMPIRank() == 0;
  }

  void InitializeEmptyData() {
    input_data_ = std::make_pair(Matrix{.rows = 0, .cols = 0, .data = {}}, Matrix{.rows = 0, .cols = 0, .data = {}});

    output_data_ = {.rows = 0, .cols = 0, .data = {}};
  }

  void LoadTestData() {
    input_data_.first.rows = kSize_;
    input_data_.first.cols = kSize_;
    input_data_.first.data.resize(kSize_ * kSize_);

    for (size_t i = 0; i < kSize_; i++) {
      for (size_t j = 0; j < kSize_; j++) {
        input_data_.first.data[(i * kSize_) + j] = static_cast<int>(i + 1);
      }
    }

    input_data_.second.rows = kSize_;
    input_data_.second.cols = kSize_;
    input_data_.second.data.resize(kSize_ * kSize_);

    for (size_t i = 0; i < kSize_; i++) {
      for (size_t j = 0; j < kSize_; j++) {
        input_data_.second.data[(i * kSize_) + j] = static_cast<int>(j + 1);
      }
    }

    output_data_.rows = kSize_;
    output_data_.cols = kSize_;
    output_data_.data.resize(kSize_ * kSize_);

    for (size_t i = 0; i < kSize_; i++) {
      for (size_t j = 0; j < kSize_; j++) {
        output_data_.data[(i * kSize_) + j] = static_cast<int>((i + 1) * (j + 1) * kSize_);
      }
    }
  }
};

TEST_P(GutyanskyAMatrixBandMultiplicationPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GutyanskyAMatrixBandMultiplicationMPI, GutyanskyAMatrixBandMultiplicationSEQ>(
        PPC_SETTINGS_gutyansky_a_matrix_band_multiplication);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GutyanskyAMatrixBandMultiplicationPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GutyanskyAMatrixBandMultiplicationPerfTest, kGtestValues, kPerfTestName);

}  // namespace gutyansky_a_matrix_band_multiplication
