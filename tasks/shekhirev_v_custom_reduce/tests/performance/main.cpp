#include <gtest/gtest.h>
#include <mpi.h>

#include "../../common/include/common.hpp"
#include "../../mpi/include/ops_mpi.hpp"
#include "../../seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace shekhirev_v_custom_reduce {

class ShekhirevVCustomReducePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_data_ = 100000;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    if (rank == 0) {
      return output_data == input_data_;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(ShekhirevVCustomReducePerfTests, PerfTest) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, shekhirev_v_custom_reduce_mpi::CustomReduceMPI,
                                                       shekhirev_v_custom_reduce_seq::CustomReduceSequential>(
    PPC_SETTINGS_shekhirev_v_custom_reduce);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShekhirevVCustomReducePerfTests::CustomPerfTestName;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(CustomReducePerf, ShekhirevVCustomReducePerfTests, kGtestValues, kPerfTestName);

}  // namespace
}  // namespace shekhirev_v_custom_reduce
