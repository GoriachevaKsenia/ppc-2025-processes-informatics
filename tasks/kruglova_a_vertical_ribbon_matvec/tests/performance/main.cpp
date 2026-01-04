#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "kruglova_a_vertical_ribbon_matvec/common/include/common.hpp"
#include "kruglova_a_vertical_ribbon_matvec/mpi/include/ops_mpi.hpp"
#include "kruglova_a_vertical_ribbon_matvec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kruglova_a_vertical_ribbon_matvec {

class KruglovaAVerticalRibMatPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int k_sizen_ = 6000;
  const int k_sizem_ = 6000;
  InType input_data_;

  void SetUp() override {
    const int m = k_sizem_;
    const int n = k_sizen_;

    std::vector<double> a_matrix(static_cast<size_t>(m) * n);
    std::vector<double> b_vector(n);

    for (double &x : a_matrix) {
      x = 1.0;
    }

    for (double &x : b_vector) {
      x = 1.0;
    }

    input_data_ = std::make_tuple(m, n, a_matrix, b_vector);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const int m = k_sizem_;
    return output_data.size() == static_cast<std::size_t>(m);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KruglovaAVerticalRibMatPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KruglovaAVerticalRibbMatMPI, KruglovaAVerticalRibbMatSEQ>(
        PPC_SETTINGS_kruglova_a_vertical_ribbon_matvec);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KruglovaAVerticalRibMatPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KruglovaAVerticalRibMatPerfTests, kGtestValues, kPerfTestName);

}  // namespace kruglova_a_vertical_ribbon_matvec
