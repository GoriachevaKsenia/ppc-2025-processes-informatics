#include <gtest/gtest.h>
#include <mpi.h>

#include <vector>

#include "lifanov_k_allreduce/common/include/common.hpp"
#include "lifanov_k_allreduce/mpi/include/ops_mpi.hpp"
#include "lifanov_k_allreduce/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace lifanov_k_allreduce {

class LifanovKAllReducePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int size_ = 20000000;
  InType input_data_;

  void SetUp() override {
    input_data_ = std::vector<int>(size_, 1);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
      return true;
    }
    return output_data.size() == 1;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LifanovKAllReducePerfTests, MyRunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LifanovKAllReduceMPI, LifanovKAllreduceSEQ>(PPC_SETTINGS_lifanov_k_allreduce);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LifanovKAllReducePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(MyAllReduceTests, LifanovKAllReducePerfTests, kGtestValues, kPerfTestName);

}  // namespace lifanov_k_allreduce
