#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <random>

#include "akimov_i_radix_sort_double_batcher_merge/common/include/common.hpp"
#include "akimov_i_radix_sort_double_batcher_merge/mpi/include/ops_mpi.hpp"
#include "akimov_i_radix_sort_double_batcher_merge/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace akimov_i_radix_sort_double_batcher_merge {

class AkimovIRunPerfTestRadixBatcherSort : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 1000000;
  InType input_data_;

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1e6, 1e6);

    input_data_.resize(kCount_);
    for (int i = 0; i < kCount_; ++i) {
      input_data_[i] = dist(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    OutType expected_data = input_data_;
    std::ranges::sort(expected_data);
    if (output_data.size() != expected_data.size()) {
      return false;
    }
    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_data[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(AkimovIRunPerfTestRadixBatcherSort, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, AkimovIRadixBatcherSortMPI, AkimovIRadixBatcherSortSEQ>(
    PPC_SETTINGS_akimov_i_radix_sort_double_batcher_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = AkimovIRunPerfTestRadixBatcherSort::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, AkimovIRunPerfTestRadixBatcherSort, kGtestValues, kPerfTestName);

}  // namespace akimov_i_radix_sort_double_batcher_merge
