#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>

#include "gasenin_l_mult_int_mstep_trapez/common/include/common.hpp"
#include "gasenin_l_mult_int_mstep_trapez/mpi/include/ops_mpi.hpp"
#include "gasenin_l_mult_int_mstep_trapez/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace gasenin_l_mult_int_mstep_trapez {

class GaseninLIntegralPerformanceTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_data_ = TaskData{.n_steps = 2000, .func_id = 1, .x1 = 0.0, .x2 = 1.0, .y1 = 0.0, .y2 = 1.0};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) {
      return true;
    }

    double expected = 2.0 / 3.0;
    double tolerance = 1e-3;
    return std::abs(output_data - expected) < tolerance;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

TEST_P(GaseninLIntegralPerformanceTests, PerformanceTests) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GaseninLMultIntMstepTrapezMPI, GaseninLMultIntMstepTrapezSEQ>(
        PPC_SETTINGS_gasenin_l_mult_int_mstep_trapez);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = GaseninLIntegralPerformanceTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(GaseninLPerformanceTestSuite, GaseninLIntegralPerformanceTests, kGtestValues, kPerfTestName);

}  // namespace gasenin_l_mult_int_mstep_trapez
