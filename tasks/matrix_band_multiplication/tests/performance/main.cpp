#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "matrix_band_multiplication/common/include/common.hpp"
#include "matrix_band_multiplication/mpi/include/ops_mpi.hpp"
#include "matrix_band_multiplication/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace matrix_band_multiplication {

class MatrixBandMultiplicationPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    constexpr std::size_t kRowsA = 640;
    constexpr std::size_t kColsA = 640;
    constexpr std::size_t kColsB = 640;

    input_.a.rows = kRowsA;
    input_.a.cols = kColsA;
    input_.a.values.resize(kRowsA * kColsA);
    for (std::size_t i = 0; i < kRowsA; ++i) {
      for (std::size_t j = 0; j < kColsA; ++j) {
        input_.a.values[FlattenIndex(i, j, kColsA)] = static_cast<double>((i + j) % 17) - 8.0;
      }
    }

    input_.b.rows = kColsA;
    input_.b.cols = kColsB;
    input_.b.values.resize(kColsA * kColsB);
    for (std::size_t i = 0; i < kColsA; ++i) {
      for (std::size_t j = 0; j < kColsB; ++j) {
        input_.b.values[FlattenIndex(i, j, kColsB)] = static_cast<double>((i * 3 + j) % 23) * 0.1;
      }
    }

    reference_ = ComputeReference();
  }

  InType GetTestInputData() final {
    return input_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.rows != reference_.rows || output_data.cols != reference_.cols) {
      return false;
    }
    const double eps = 1e-6;
    for (std::size_t idx = 0; idx < reference_.values.size(); ++idx) {
      if (std::fabs(reference_.values[idx] - output_data.values[idx]) > eps) {
        return false;
      }
    }
    return true;
  }

 private:
  [[nodiscard]] Matrix ComputeReference() const {
    Matrix result;
    result.rows = input_.a.rows;
    result.cols = input_.b.cols;
    result.values.assign(result.rows * result.cols, 0.0);
    for (std::size_t i = 0; i < input_.a.rows; ++i) {
      for (std::size_t j = 0; j < input_.b.cols; ++j) {
        double sum = 0.0;
        for (std::size_t k = 0; k < input_.a.cols; ++k) {
          sum +=
              input_.a.values[FlattenIndex(i, k, input_.a.cols)] * input_.b.values[FlattenIndex(k, j, input_.b.cols)];
        }
        result.values[FlattenIndex(i, j, result.cols)] = sum;
      }
    }
    return result;
  }

  InType input_{};
  Matrix reference_{};
};

namespace {

const auto kPerfTasks = ppc::util::MakeAllPerfTasks<InType, MatrixBandMultiplicationMpi, MatrixBandMultiplicationSeq>(
    PPC_SETTINGS_matrix_band_multiplication);

const auto kGTestValues = ppc::util::TupleToGTestValues(kPerfTasks);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(MatrixProductPerf, MatrixBandMultiplicationPerfTests, kGTestValues,
                         MatrixBandMultiplicationPerfTests::CustomPerfTestName);

TEST_P(MatrixBandMultiplicationPerfTests, Runs) {
  ExecuteTest(GetParam());
}

}  // namespace

}  // namespace matrix_band_multiplication
