#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

#include "frolova_s_sum_elem_matrix/common/include/common.hpp"
#include "frolova_s_sum_elem_matrix/mpi/include/ops_mpi.hpp"
#include "frolova_s_sum_elem_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace frolova_s_sum_elem_matrix {

class FrolovaSSumElemMatrixRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  const int kCount_ = 10000000;
  OutType expected_res_ = 49999995000000.0;

  void SetUp() override {
    int vect_size = kCount_;
    std::vector<double> vect_data(vect_size);

    for (int i = 0; i < vect_size; i++) {
      vect_data[i] = static_cast<double>(i);
    }

    expected_res_ = 0;
    for (double val : vect_data) {
      expected_res_ += val;
    }

    input_data_ = std::make_tuple(vect_data, vect_size, 1);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data == std::numeric_limits<double>::max()) {
      return true;
    }

    return std::abs(output_data - expected_res_) < 1e-7;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, FrolovaSSumElemMatrixMPI, FrolovaSSumElemMatrixSEQ>(
    PPC_SETTINGS_frolova_s_sum_elem_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = FrolovaSSumElemMatrixRunPerfTestsProcesses::CustomPerfTestName;

TEST_P(FrolovaSSumElemMatrixRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(RunPerf, FrolovaSSumElemMatrixRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace frolova_s_sum_elem_matrix
