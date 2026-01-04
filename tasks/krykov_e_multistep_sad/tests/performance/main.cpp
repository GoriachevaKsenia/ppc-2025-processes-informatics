#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <tuple>

#include "krykov_e_multistep_sad/common/include/common.hpp"
#include "krykov_e_multistep_sad/mpi/include/ops_mpi.hpp"
#include "krykov_e_multistep_sad/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace krykov_e_multistep_sad {

class KrykovEMultistepSADPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    input_data_ = {[](double x, double y) {
      double s = 0.0;
      for (int i = 1; i <= 5000; ++i) {
        const double dx = x - 1.0;
        const double dy = y + 2.0;
        s += (dx * dx) + (dy * dy) + static_cast<double>(i);
      }
      return s;
    }, -10.0, 10.0, -10.0, 10.0};

    const double expected_value = 5000 * (5000 + 1) / 2.0;

    expected_output_ = {1.0, -2.0, expected_value};
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
  KrykovEMultistepSADPerfTests() = default;
};

TEST_P(KrykovEMultistepSADPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KrykovEMultistepSADMPI, KrykovEMultistepSADSEQ>(
    PPC_SETTINGS_krykov_e_multistep_sad);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KrykovEMultistepSADPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KrykovEMultistepSADPerfTests, kGtestValues, kPerfTestName);

}  // namespace krykov_e_multistep_sad
