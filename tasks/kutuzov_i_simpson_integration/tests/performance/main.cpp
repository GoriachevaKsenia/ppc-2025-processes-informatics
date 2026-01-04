#include <gtest/gtest.h>

#include <cmath>
#include <tuple>

#include "kutuzov_i_simpson_integration/common/include/common.hpp"
#include "kutuzov_i_simpson_integration/mpi/include/ops_mpi.hpp"
#include "kutuzov_i_simpson_integration/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kutuzov_i_simpson_integration {

class KutuzovIRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    input_data_ = {500, {-10.0, 10.0}, {-10.0, 10.0}, 4};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int n = std::get<0>(input_data_);
    double x_min = std::get<1>(input_data_).first;
    double x_max = std::get<1>(input_data_).second;
    double y_min = std::get<2>(input_data_).first;
    double y_max = std::get<2>(input_data_).second;
    int function_id = std::get<3>(input_data_);

    double step_x = (x_max - x_min) / n;
    double step_y = (y_max - y_min) / n;

    double sum = 0;
    for (int i = 0; i <= n; i++) {
      double x = x_min + (step_x * i);

      for (int j = 0; j <= n; j++) {
        double y = y_min + (step_y * j);
        sum += GetWeight(i, n) * GetWeight(j, n) * CallFunction(function_id, x, y);
      }
    }
    sum *= step_x * step_y / 9;
    return (abs(output_data - sum) < 1e-3);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  static double GetWeight(int i, int n) {
    if (i == 0 || i == n) {
      return 1.0;
    }
    if (i % 2 == 1) {
      return 4.0;
    }
    return 2.0;
  }
};

TEST_P(KutuzovIRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KutuzovISimpsonIntegrationMPI, KutuzovISimpsonIntegrationSEQ>(
        PPC_SETTINGS_kutuzov_i_simpson_integration);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KutuzovIRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KutuzovIRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace kutuzov_i_simpson_integration
