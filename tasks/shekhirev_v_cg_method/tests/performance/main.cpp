#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "../../common/include/common.hpp"
#include "../../mpi/include/ops_mpi.hpp"
#include "../../seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace shekhirev_v_cg_method {

class ShekhirevVCGPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    int n = 10000;
    input_data_.n = n;
    input_data_.A.resize(static_cast<size_t>(n) * n, 0.0);
    input_data_.b.resize(n, 2.0);

    for (int i = 0; i < n; ++i) {
      input_data_.A[(i * n) + i] = 2.0;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    if (rank == 0) {
      if (output_data.empty()) {
        return false;
      }
      for (double val : output_data) {
        if (std::abs(val - 1.0) > 1e-4) {
          return false;
        }
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(ShekhirevVCGPerfTests, PerfTest) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, shekhirev_v_cg_method_mpi::ConjugateGradientMPI,
                                shekhirev_v_cg_method_seq::ConjugateGradientSeq>(PPC_SETTINGS_shekhirev_v_cg_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = ShekhirevVCGPerfTests::CustomPerfTestName;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(CGPerf, ShekhirevVCGPerfTests, kGtestValues, kPerfTestName);

}  // namespace
}  // namespace shekhirev_v_cg_method
