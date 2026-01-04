#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "romanova_v_jacobi_method/common/include/common.hpp"
#include "romanova_v_jacobi_method/mpi/include/ops_mpi.hpp"
#include "romanova_v_jacobi_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace romanova_v_jacobi_method {

class RomanovaVJacobiMethodPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    size_t n = 200;
    std::vector<std::vector<double>> a(n, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < n; j++) {
        if (i == j) {
          a[i][j] = 10.01;
        }
        if (j == i - 1 || j == i + 1) {
          a[i][j] = 5.0;
        }
      }
    }

    std::vector<double> x(n, -1000.0);
    std::vector<double> b(n, 20.01);
    b[0] = b[n - 1] = 15.01;
    size_t iterations = 100000;
    eps_ = 1e-9;

    input_data_ = std::make_tuple(x, a, b, eps_, iterations);
    exp_answer_ = std::vector<double>(n, 1.0);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != exp_answer_.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); i++) {
      if (abs(output_data[i] - exp_answer_[i]) > eps_) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType exp_answer_;
  double eps_ = 0.0;
};

TEST_P(RomanovaVJacobiMethodPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, RomanovaVJacobiMethodMPI, RomanovaVJacobiMethodSEQ>(
    PPC_SETTINGS_romanova_v_jacobi_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RomanovaVJacobiMethodPerfTestProcesses::CustomPerfTestName;
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits, misc-use-anonymous-namespace)
INSTANTIATE_TEST_SUITE_P(RunModeTests, RomanovaVJacobiMethodPerfTestProcesses, kGtestValues, kPerfTestName);
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits, misc-use-anonymous-namespace)
}  // namespace romanova_v_jacobi_method
