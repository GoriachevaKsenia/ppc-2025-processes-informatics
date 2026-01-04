#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "krykov_e_simple_iterations/common/include/common.hpp"
#include "krykov_e_simple_iterations/mpi/include/ops_mpi.hpp"
#include "krykov_e_simple_iterations/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace krykov_e_simple_iterations {

class KrykovESimpleIterationsPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    constexpr size_t kN = 200;

    std::vector<double> a(kN * kN, 0.0);
    std::vector<double> b(kN, 0.0);
    expected_output_.assign(kN, 1.0);

    for (size_t i = 0; i < kN; ++i) {
      double row_sum = 0.0;
      for (size_t j = 0; j < kN; ++j) {
        if (i != j) {
          a[(i * kN) + j] = 1.0;
          row_sum += 1.0;
        }
      }
      a[(i * kN) + i] = row_sum + 5.0;
      b[i] = a[(i * kN) + i] + row_sum;
    }

    input_data_ = std::make_tuple(kN, a, b);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    constexpr double kEps = 1e-4;
    if (output_data.size() != expected_output_.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_output_[i]) > kEps) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 public:
  KrykovESimpleIterationsPerfTests() = default;
};

TEST_P(KrykovESimpleIterationsPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KrykovESimpleIterationsMPI, KrykovESimpleIterationsSEQ>(
    PPC_SETTINGS_krykov_e_simple_iterations);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KrykovESimpleIterationsPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KrykovESimpleIterationsPerfTests, kGtestValues, kPerfTestName);

}  // namespace krykov_e_simple_iterations
