#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "dolov_v_monte_carlo_integration/common/include/common.hpp"
#include "dolov_v_monte_carlo_integration/mpi/include/ops_mpi.hpp"
#include "dolov_v_monte_carlo_integration/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dolov_v_monte_carlo_integration {
namespace {

class DolovVMonteCarloIntegrationPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  static constexpr int kSamples = 1000000;
  InType input_data_{};
  OutType expected_result_ = 0.0;

  static double FuncSumCoords3D(const std::vector<double> &x) {
    double sum_val = 0.0;
    for (double val : x) {
      sum_val += val;
    }
    return sum_val;
  }

 protected:
  void SetUp() override {
    const int k_dim = 3;
    const double k_rad = 1.0;
    const std::vector<double> k_center = {0.0, 0.0, 0.0};

    input_data_ = {.func = FuncSumCoords3D,
                   .dimension = k_dim,
                   .samples_count = kSamples,
                   .center = k_center,
                   .radius = k_rad,
                   .domain_type = IntegrationDomain::kHyperCube};
    expected_result_ = 0.0;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::abs(output_data - expected_result_) < 0.05;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(DolovVMonteCarloIntegrationPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, DolovVMonteCarloIntegrationMPI, DolovVMonteCarloIntegrationSEQ>(
        PPC_SETTINGS_dolov_v_monte_carlo_integration);

const auto kPerfVals = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfName = DolovVMonteCarloIntegrationPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(MonteCarlo3DPerf, DolovVMonteCarloIntegrationPerfTests, kPerfVals, kPerfName);

}  // namespace
}  // namespace dolov_v_monte_carlo_integration
