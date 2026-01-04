#include <gtest/gtest.h>

#include <cmath>
#include <tuple>

#include "eremin_v_strongin_algorithm/common/include/common.hpp"
#include "eremin_v_strongin_algorithm/mpi/include/ops_mpi.hpp"
#include "eremin_v_strongin_algorithm/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace eremin_v_strongin_algorithm {

class EreminVRunPerfTestsStronginAlgorithm : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    double lower_bound = -5.0;
    double upper_bound = 5.0;
    double epsilon = 1e-4;
    int max_iters = 40000;
    input_data_ = std::make_tuple(lower_bound, upper_bound, epsilon, max_iters, Function);
    expected_result_ = -6.1;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double tolerance = 1e-2;
    return std::abs(output_data - expected_result_) <= tolerance;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_result_{};

  static double Function(double x) {
    return (0.002 * x * x) + (5.0 * std::sin(30.0 * x)) + std::sin(200.0 * std::sin(50.0 * x)) +
           (0.1 * std::cos(300.0 * x));
    ;
  }
};

TEST_P(EreminVRunPerfTestsStronginAlgorithm, RunPerfModesStronginAlgorithm) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, EreminVStronginAlgorithmMPI, EreminVStronginAlgorithmSEQ>(
        PPC_SETTINGS_eremin_v_strongin_algorithm);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = EreminVRunPerfTestsStronginAlgorithm::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTestsStronginAlgorithm, EreminVRunPerfTestsStronginAlgorithm, kGtestValues,
                         kPerfTestName);

}  // namespace eremin_v_strongin_algorithm
