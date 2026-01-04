#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <tuple>
#include <vector>

#include "romanov_a_scatter/common/include/common.hpp"
#include "romanov_a_scatter/mpi/include/ops_mpi.hpp"
#include "romanov_a_scatter/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace romanov_a_scatter {

class RomanovAScatterPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 150'000'000;
  InType input_data_;

  bool static IsSeqTest() {
    const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    if (test_info == nullptr) {
      return false;
    }
    return std::string(test_info->name()).find("seq") != std::string::npos;
  }

  void SetUp() override {
    int rank = 0;
    int num_processes = 1;

    bool is_seq_test = IsSeqTest();

    if (!is_seq_test) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    }

    int sendcount = (kCount_ + num_processes - 1) / num_processes;
    int root = num_processes / 2;

    std::vector<int> sendbuf;
    if (rank == root) {
      int n = sendcount * num_processes;
      sendbuf.resize(n);
      for (int i = 0; i < n; ++i) {
        sendbuf[i] = i;
      }
    }

    input_data_ = std::make_tuple(sendbuf, sendcount, root);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(RomanovAScatterPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RomanovAScatterMPI, RomanovAScatterSEQ>(PPC_SETTINGS_romanov_a_scatter);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RomanovAScatterPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RomanovAScatterPerfTests, kGtestValues, kPerfTestName);

}  // namespace romanov_a_scatter
