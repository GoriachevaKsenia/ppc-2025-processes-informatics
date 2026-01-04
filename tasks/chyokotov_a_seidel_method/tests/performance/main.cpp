#include <gtest/gtest.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <vector>

#include "chyokotov_a_seidel_method/common/include/common.hpp"
#include "chyokotov_a_seidel_method/mpi/include/ops_mpi.hpp"
#include "chyokotov_a_seidel_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace chyokotov_a_seidel_method {

class ChyokotovASeidelMethodPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    const int n = 5000;
    auto &matrix = input_data_.first;
    auto &vector = input_data_.second;
    matrix.resize(n, std::vector<double>(n, 1.0));
    vector.resize(n);
    expected_output_.resize(n);

    for (int i = 0; i < n; i++) {
      expected_output_[i] = 1.0 + (static_cast<double>(i) / 5000.0);
      matrix[i][i] = 5001.0;
    }

    for (int i = 0; i < n; i++) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        sum += matrix[i][j] * expected_output_[j];
      }
      vector[i] = sum;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_output_[i]) > 0.0001) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ChyokotovASeidelMethodPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ChyokotovASeidelMethodMPI, ChyokotovASeidelMethodSEQ>(
    PPC_SETTINGS_chyokotov_a_seidel_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ChyokotovASeidelMethodPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ChyokotovASeidelMethodPerfTest, kGtestValues, kPerfTestName);

}  // namespace chyokotov_a_seidel_method
