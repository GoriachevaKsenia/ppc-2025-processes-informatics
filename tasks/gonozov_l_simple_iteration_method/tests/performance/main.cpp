#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "gonozov_l_simple_iteration_method/common/include/common.hpp"
#include "gonozov_l_simple_iteration_method/mpi/include/ops_mpi.hpp"
#include "gonozov_l_simple_iteration_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace gonozov_l_simple_iteration_method {

class GonozovLRunIterationMethodPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 600;
  InType input_data_;
  OutType desired_result_;

  void SetUp() override {
    std::vector<double> matrix(static_cast<size_t>(kCount_ * kCount_), 1.0);
    std::vector<double> b(kCount_, 1.0);

    for (int i = 0; i < kCount_; i++) {
      matrix[(i * kCount_) + i] = static_cast<double>(kCount_);
      b[i] = static_cast<double>((kCount_ * 2) - 1);
    }

    input_data_ = std::make_tuple(kCount_, matrix, b);
    std::vector<double> desired(kCount_, 1.0);
    desired_result_ = desired;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (desired_result_.size() != output_data.size()) {
      return false;
    }
    for (unsigned int i = 0; i < desired_result_.size(); i++) {
      if (std::abs(desired_result_[i] - output_data[i]) > 0.01) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(GonozovLRunIterationMethodPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GonozovLSimpleIterationMethodMPI, GonozovLSimpleIterationMethodSEQ>(
        PPC_SETTINGS_gonozov_l_simple_iteration_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GonozovLRunIterationMethodPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GonozovLRunIterationMethodPerfTest, kGtestValues, kPerfTestName);

}  // namespace gonozov_l_simple_iteration_method
