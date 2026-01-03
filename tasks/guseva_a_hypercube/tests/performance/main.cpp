#include <gtest/gtest.h>

#include <cmath>

#include "guseva_a_hypercube/common/include/common.hpp"
#include "guseva_a_hypercube/mpi/include/ops_mpi.hpp"
#include "util/include/perf_test_util.hpp"

namespace guseva_a_hypercube {

namespace {
class GusevaAHypercubePerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 1e8;
  InType input_data_;
  OutType output_data_ = 0;

  void SetUp() override {
    input_data_ = {1, kCount_, [](int x) { return std::sin(x); }, "perf"};
    output_data_ = (std::sin(kCount_) - std::cos(kCount_) / std::tan(1. / 2) + 1. / std::tan(1. / 2)) / 2;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double precision = 1e-6;
    return output_data_ - precision <= output_data && output_data <= output_data_ + precision;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(GusevaAHypercubePerfTestProcesses, GusevaAHypercubePerf) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, GusevaAHypercubeMine>(PPC_SETTINGS_guseva_a_hypercube);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GusevaAHypercubePerfTestProcesses::CustomPerfTestName;

// NOLINTNEXTLINE // To avoid "Variable <...> is non-cost and globally accessible, consider making it const"
INSTANTIATE_TEST_SUITE_P(GusevaAHypercubePerf, GusevaAHypercubePerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace guseva_a_hypercube
