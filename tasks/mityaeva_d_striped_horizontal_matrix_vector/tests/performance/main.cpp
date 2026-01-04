#include <gtest/gtest.h>

#include <cstddef>

#include "mityaeva_d_striped_horizontal_matrix_vector/common/include/common.hpp"
#include "mityaeva_d_striped_horizontal_matrix_vector/mpi/include/ops_mpi.hpp"
#include "mityaeva_d_striped_horizontal_matrix_vector/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace mityaeva_d_striped_horizontal_matrix_vector {

class StripedHorizontalMatrixVectorRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixSize_ = 2000;
  InType input_data_;

  void SetUp() override {
    int rows = kMatrixSize_;
    int cols = kMatrixSize_;

    input_data_.clear();
    input_data_.reserve(3 + (rows * cols) + cols);

    input_data_.push_back(rows);
    input_data_.push_back(cols);
    input_data_.push_back(cols);

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        input_data_.push_back(static_cast<double>((i + j) % 100) * 0.1);
      }
    }

    for (int j = 0; j < cols; ++j) {
      input_data_.push_back(static_cast<double>((j % 50) + 1) * 0.5);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return false;
    }

    int result_size = static_cast<int>(output_data[0]);
    if (result_size != kMatrixSize_) {
      return false;
    }

    if (output_data.size() != static_cast<size_t>(result_size) + 1) {
      return false;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(StripedHorizontalMatrixVectorRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, StripedHorizontalMatrixVectorMPI, StripedHorizontalMatrixVectorSEQ>(
        PPC_SETTINGS_mityaeva_d_striped_horizontal_matrix_vector);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = StripedHorizontalMatrixVectorRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, StripedHorizontalMatrixVectorRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace mityaeva_d_striped_horizontal_matrix_vector
