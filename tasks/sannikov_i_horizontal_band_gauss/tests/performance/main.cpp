#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "sannikov_i_horizontal_band_gauss/common/include/common.hpp"
#include "sannikov_i_horizontal_band_gauss/mpi/include/ops_mpi.hpp"
#include "sannikov_i_horizontal_band_gauss/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sannikov_i_horizontal_band_gauss {

class SannikIHorizontalBandGaussPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    const std::size_t n = 6000;
    const std::size_t band = 6;

    std::vector<std::vector<double>> a(n, std::vector<double>(n, 0.0));

    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t j_begin = (i > band) ? (i - band) : 0;
      const std::size_t j_end = std::min(n, i + band + 1);

      double row_abs_sum = 0.0;
      for (std::size_t j = j_begin; j < j_end; ++j) {
        if (j == i) {
          continue;
        }
        a[i][j] = 1.0;
        row_abs_sum += std::abs(a[i][j]);
      }

      a[i][i] = row_abs_sum + 10.0;
    }

    std::vector<double> x_true(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      x_true[i] = static_cast<double>(i + 1);
    }
    std::vector<double> b(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      double s = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        s += a[i][j] * x_true[j];
      }
      b[i] = s;
    }

    input_data_ = std::make_tuple(std::move(a), std::move(b), band);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &a = std::get<0>(input_data_);
    const auto &b = std::get<1>(input_data_);

    if (a.empty() || b.empty()) {
      return false;
    }
    if (output_data.size() != a.size()) {
      return false;
    }

    const std::size_t n = a.size();
    double max_residual = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
      double s = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        s += a[i][j] * output_data[j];
      }
      double res = 0.0;
      res = std::abs(s - b[i]);
      max_residual = std::max(max_residual, res);
    }

    return max_residual < 1e-7;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};
//
TEST_P(SannikIHorizontalBandGaussPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SannikovIHorizontalBandGaussMPI, SannikovIHorizontalBandGaussSEQ>(
        PPC_SETTINGS_sannikov_i_horizontal_band_gauss);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SannikIHorizontalBandGaussPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SannikIHorizontalBandGaussPerfTests, kGtestValues, kPerfTestName);

}  // namespace sannikov_i_horizontal_band_gauss
