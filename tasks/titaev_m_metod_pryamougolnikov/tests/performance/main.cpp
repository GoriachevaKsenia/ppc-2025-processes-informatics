#include <gtest/gtest.h>

#include <cmath>

#include "titaev_m_metod_pryamougolnikov/common/include/common.hpp"
#include "titaev_m_metod_pryamougolnikov/mpi/include/ops_mpi.hpp"
#include "titaev_m_metod_pryamougolnikov/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace titaev_m_metod_pryamougolnikov {

class RectanglePerformanceTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    test_input_.left_bounds = {0.0, 0.0, 0.0};
    test_input_.right_bounds = {1.0, 1.0, 1.0};
    test_input_.partitions = 20;
  }

  bool CheckTestOutputData(OutType &output_value) final {
    const double exact_value = 1.5;
    return std::abs(output_value - exact_value) < 1e-4;
  }

  InType GetTestInputData() final {
    return test_input_;
  }

 private:
  InType test_input_;
};

TEST_P(RectanglePerformanceTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kPerformanceTasks =
    ppc::util::MakeAllPerfTasks<InType, TitaevMMetodPryamougolnikovMPI, TitaevMMetodPryamougolnikovSEQ>(
        PPC_SETTINGS_titaev_m_metod_pryamougolnikov);

const auto kGtestValues = ppc::util::TupleToGTestValues(kPerformanceTasks);

INSTANTIATE_TEST_SUITE_P(PerformanceTests, RectanglePerformanceTests, kGtestValues,
                         RectanglePerformanceTests::CustomPerfTestName);

}  // namespace titaev_m_metod_pryamougolnikov
