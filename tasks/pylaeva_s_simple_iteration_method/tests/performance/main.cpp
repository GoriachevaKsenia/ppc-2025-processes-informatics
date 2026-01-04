#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "pylaeva_s_simple_iteration_method/common/include/common.hpp"
#include "pylaeva_s_simple_iteration_method/mpi/include/ops_mpi.hpp"
#include "pylaeva_s_simple_iteration_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace pylaeva_s_simple_iteration_method {

class PylaevaSSimpleIterationMethodPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;
  OutType expected_data_;

  void SetUp() override {
    constexpr size_t kN = 300;

    std::vector<double> a(kN * kN, 1.0);
    std::vector<double> b(kN, 0.0);
    expected_data_.assign(kN, 1.0);

    for (size_t i = 0; i < kN; ++i) {
      a[(i * kN) + i] = static_cast<double>(kN);
      b[i] = (2.0 * static_cast<double>(kN)) - 1.0;
    }

    input_data_ = std::make_tuple(kN, a, b);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    constexpr double kEps = 1e-6;
    if (output_data.size() != expected_data_.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_data_[i]) > kEps) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 public:
  PylaevaSSimpleIterationMethodPerfTests() = default;
};

TEST_P(PylaevaSSimpleIterationMethodPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, PylaevaSSimpleIterationMethodMPI, PylaevaSSimpleIterationMethodSEQ>(
        PPC_SETTINGS_pylaeva_s_simple_iteration_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PylaevaSSimpleIterationMethodPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PylaevaSSimpleIterationMethodPerfTests, kGtestValues, kPerfTestName);

}  // namespace pylaeva_s_simple_iteration_method
