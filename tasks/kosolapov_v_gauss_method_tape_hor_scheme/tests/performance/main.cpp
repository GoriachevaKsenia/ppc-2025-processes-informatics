#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "kosolapov_v_gauss_method_tape_hor_scheme/common/include/common.hpp"
#include "kosolapov_v_gauss_method_tape_hor_scheme/mpi/include/ops_mpi.hpp"
#include "kosolapov_v_gauss_method_tape_hor_scheme/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kosolapov_v_gauss_method_tape_hor_scheme {

class KosolapovVGaussMethodTapeHorSchemeRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixSize_ = 500;
  InType input_data_;

  void SetUp() override {
    InType system;
    system.matrix.resize(kMatrixSize_, std::vector<double>(kMatrixSize_));
    system.r_side.resize(kMatrixSize_);
    for (int i = 0; i < kMatrixSize_; i++) {
      double row_sum = 0.0;
      for (int j = 0; j < kMatrixSize_; j++) {
        if (i != j) {
          system.matrix[i][j] = 1.0 / (i + j + 2);
          row_sum += system.matrix[i][j];
        }
      }
      system.matrix[i][i] = row_sum + ((i + 1) * 2.0);
      system.r_side[i] = row_sum + system.matrix[i][i];
    }
    input_data_ = system;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &matrix = input_data_.matrix;
    const auto &r_side = input_data_.r_side;
    int n = static_cast<int>(matrix.size());
    double epsilon = 1e-6;
    bool solution_correct = true;

    for (int i = 0; i < n; i++) {
      double sum = 0.0;
      for (int j = 0; j < n; j++) {
        sum += matrix[i][j] * output_data[j];
      }
      double diff = std::abs(sum - r_side[i]);
      if (diff > epsilon) {
        solution_correct = false;
      }
    }
    return solution_correct;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KosolapovVGaussMethodTapeHorSchemeRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KosolapovVGaussMethodTapeHorSchemeMPI, KosolapovVGaussMethodTapeHorSchemeSEQ>(
        PPC_SETTINGS_kosolapov_v_gauss_method_tape_hor_scheme);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KosolapovVGaussMethodTapeHorSchemeRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunGaussPerfTests, KosolapovVGaussMethodTapeHorSchemeRunPerfTestProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace kosolapov_v_gauss_method_tape_hor_scheme
