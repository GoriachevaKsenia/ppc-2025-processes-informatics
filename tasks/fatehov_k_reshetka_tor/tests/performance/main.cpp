#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include "fatehov_k_reshetka_tor/common/include/common.hpp"
#include "fatehov_k_reshetka_tor/mpi/include/ops_mpi.hpp"
#include "fatehov_k_reshetka_tor/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace fatehov_k_reshetka_tor {

class FatehovKRunPerfTestsReshetkaTor : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_ = std::make_tuple(0, 0, std::vector<double>{});
  OutType expected_result_ = 0;

  void SetUp() override {
    const size_t rows = 1000;
    const size_t cols = 1000;
    const size_t total = rows * cols;

    uint64_t state = 42;
    const uint64_t a = 1664525ULL;
    const uint64_t c = 1013904223ULL;
    const uint64_t m = (1ULL << 22);

    std::vector<double> matrix;
    matrix.reserve(total);

    for (size_t i = 0; i < total; ++i) {
      state = (a * state + c) % m;
      double value = ((static_cast<double>(state) / m) * 2000.0) - 1000.0;
      matrix.push_back(value);
    }

    double computed_max = -std::numeric_limits<double>::max();
    for (size_t i = 0; i < total; ++i) {
      double heavy_val = matrix[i];
      for (int k = 0; k < 100; ++k) {
        heavy_val = (std::sin(heavy_val) * std::cos(heavy_val)) + std::exp(std::complex<double>(0, heavy_val).real()) +
                    std::sqrt(std::abs(heavy_val) + 1.0);
        if (std::isinf(heavy_val)) {
          heavy_val = matrix[i];
        }
      }
      computed_max = std::max(heavy_val, computed_max);
    }

    expected_result_ = computed_max;
    input_data_ = std::make_tuple(rows, cols, matrix);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_result_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(FatehovKRunPerfTestsReshetkaTor, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, FatehovKReshetkaTorMPI, FatehovKReshetkaTorSEQ>(
    PPC_SETTINGS_fatehov_k_reshetka_tor);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = FatehovKRunPerfTestsReshetkaTor::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunPerfTest, FatehovKRunPerfTestsReshetkaTor, kGtestValues, kPerfTestName);

}  // namespace fatehov_k_reshetka_tor
