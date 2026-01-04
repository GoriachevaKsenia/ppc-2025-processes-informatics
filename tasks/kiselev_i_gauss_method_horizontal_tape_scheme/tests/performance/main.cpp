#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "kiselev_i_gauss_method_horizontal_tape_scheme/common/include/common.hpp"
#include "kiselev_i_gauss_method_horizontal_tape_scheme/mpi/include/ops_mpi.hpp"
#include "kiselev_i_gauss_method_horizontal_tape_scheme/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kiselev_i_gauss_method_horizontal_tape_scheme {

class KiselevIPerfTests2 : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    const std::size_t num = 6000;
    const std::size_t band = 6;

    std::vector<std::vector<double>> a_vector(num, std::vector<double>(num, 0.0));

    for (std::size_t index = 0; index < num; ++index) {
      const std::size_t j_begin = (index > band) ? (index - band) : 0;
      const std::size_t j_end = std::min(num, index + band + 1);

      double sum_abs = 0.0;
      for (std::size_t j_index = j_begin; j_index < j_end; ++j_index) {
        if (j_index == index) {
          continue;
        }

        const double val = (j_index < index) ? 0.5 : 1.5;
        a_vector[index][j_index] = val;
        sum_abs += std::abs(val);
      }

      a_vector[index][index] = sum_abs + 8.0 + static_cast<double>(index % 3);
    }

    std::vector<double> x_true(num);
    for (std::size_t index = 0; index < num; ++index) {
      x_true[index] = 1.0 + static_cast<double>((index * 3) % 11);
    }

    std::vector<double> b_vector(num, 0.0);
    for (std::size_t index = 0; index < num; ++index) {
      const std::size_t j_begin = (index > band) ? (index - band) : 0;
      const std::size_t j_end = std::min(num, index + band + 1);

      double s = 0.0;
      for (std::size_t j_index = j_begin; j_index < j_end; ++j_index) {
        s += a_vector[index][j_index] * x_true[j_index];
      }
      b_vector[index] = s;
    }

    input_data_ = std::make_tuple(std::move(a_vector), std::move(b_vector), band);
  }

  bool CheckTestOutputData(OutType &output) final {
    const auto &a_vector = std::get<0>(input_data_);
    const auto &b_vector = std::get<1>(input_data_);
    const std::size_t band = std::get<2>(input_data_);

    const std::size_t num = a_vector.size();
    if (output.size() != num) {
      return false;
    }

    double max_residual = 0.0;

    for (std::size_t index = 0; index < num; ++index) {
      double s = 0.0;

      const std::size_t j_begin = (index > band) ? (index - band) : 0;
      const std::size_t j_end = std::min(num, index + band + 1);

      for (std::size_t j_index = j_begin; j_index < j_end; ++j_index) {
        s += a_vector[index][j_index] * output[j_index];
      }

      max_residual = std::max(max_residual, std::abs(s - b_vector[index]));
    }

    return max_residual < 1e-7;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(KiselevIPerfTests2, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KiselevITestTaskMPI, KiselevITestTaskSEQ>(
    PPC_SETTINGS_kiselev_i_gauss_method_horizontal_tape_scheme);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KiselevIPerfTests2::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KiselevIPerfTests2, kGtestValues, kPerfTestName);

}  // namespace kiselev_i_gauss_method_horizontal_tape_scheme
