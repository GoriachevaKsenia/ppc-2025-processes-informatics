#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>

#include "gutyansky_a_monte_carlo_multi_dimension/common/include/common.hpp"
#include "gutyansky_a_monte_carlo_multi_dimension/mpi/include/ops_mpi.hpp"
#include "gutyansky_a_monte_carlo_multi_dimension/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace gutyansky_a_monte_carlo_multi_dimension {

class GutyanskyAMonteCarloMultiDimensionPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    if (ShouldLoadDataAndTest()) {
      LoadTestData();
      return;
    }

    InitializeEmptyData();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double eps = 1e-1;

    if (ShouldLoadDataAndTest()) {
      return std::abs(output_data - output_data_) <= eps;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  const size_t kSize_ = 8000000;
  InType input_data_ = {};
  OutType output_data_ = 0.0;

  [[nodiscard]] static bool ShouldLoadDataAndTest() {
    const std::string &test_name =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(GetParam());

    if (test_name.find("_mpi") == std::string::npos) {
      return true;
    }

    return !ppc::util::IsUnderMpirun() || ppc::util::GetMPIRank() == 0;
  }

  void InitializeEmptyData() {
    input_data_ = {};
    output_data_ = 0.0;
  }

  void LoadTestData() {
    input_data_.func_id = 6;
    input_data_.n_dims = 2;
    input_data_.n_points = kSize_;
    input_data_.lower_bounds.resize(2);
    input_data_.lower_bounds[0] = -1.0;
    input_data_.lower_bounds[1] = -1.0;
    input_data_.upper_bounds.resize(2);
    input_data_.upper_bounds[0] = 1.0;
    input_data_.upper_bounds[1] = 1.0;
    output_data_ = std::numbers::pi_v<double>;
  }
};

TEST_P(GutyanskyAMonteCarloMultiDimensionPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GutyanskyAMonteCarloMultiDimensionMPI, GutyanskyAMonteCarloMultiDimensionSEQ>(
        PPC_SETTINGS_gutyansky_a_monte_carlo_multi_dimension);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GutyanskyAMonteCarloMultiDimensionPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GutyanskyAMonteCarloMultiDimensionPerfTest, kGtestValues, kPerfTestName);

}  // namespace gutyansky_a_monte_carlo_multi_dimension
