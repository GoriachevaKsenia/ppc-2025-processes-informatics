#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <tuple>

#include "eremin_v_hypercube/common/include/common.hpp"
#include "eremin_v_hypercube/mpi/include/ops_mpi.hpp"
#include "eremin_v_hypercube/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace eremin_v_hypercube {

class EreminVRunPerfTestsHypercube : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    input_data_ = std::make_tuple(0.0, 10.0, 100000000, InFunction);
    expected_result_ = Function(10.0) - Function(0.0);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double tolerance = std::max(std::abs(expected_result_) * 0.01, 1e-8);
    return std::abs(output_data - expected_result_) <= tolerance;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_result_{};

  static double Function(double x) {
    return (x * x * std::exp(x) * std::sin(x)) + (x * x * x * x * std::cos(2 * x));
  }
  static double InFunction(double x) {
    double term1 = std::exp(x) * (x * x * std::sin(x) + 2 * x * std::sin(x) + x * x * std::cos(x));
    double term2 = (4 * x * x * x * std::cos(2 * x)) - (2 * x * x * x * x * std::sin(2 * x));
    return term1 + term2;
  }
};

TEST_P(EreminVRunPerfTestsHypercube, RunPerfModesHypercube) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, EreminVHypercubeMPI, EreminVHypercubeSEQ>(PPC_SETTINGS_eremin_v_hypercube);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = EreminVRunPerfTestsHypercube::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTestsHypercube, EreminVRunPerfTestsHypercube, kGtestValues, kPerfTestName);

}  // namespace eremin_v_hypercube
