#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "zavyalov_a_qsort_simple_merge/common/include/common.hpp"
#include "zavyalov_a_qsort_simple_merge/mpi/include/ops_mpi.hpp"
#include "zavyalov_a_qsort_simple_merge/seq/include/ops_seq.hpp"

namespace zavyalov_a_qsort_simple_merge {

class ZavyalovAQsortPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const uint64_t kCount_ = 10000000ULL;
  InType input_data_;

  void SetUp() override {
    int is_initialized = 0;
    MPI_Initialized(&is_initialized);
    if (is_initialized == 0) {
      input_data_.resize(kCount_);
      for (size_t i = 0; i < kCount_; i++) {
        input_data_[i] = static_cast<double>(((i * 8U) - 518390U) % 126659U);
      }
    } else {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if (rank == 0) {
        input_data_.resize(kCount_);
        for (size_t i = 0; i < kCount_; i++) {
          input_data_[i] = static_cast<double>(((i * 8U) - 518390U) % 126659U);
        }
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int is_initialized = 0;
    MPI_Initialized(&is_initialized);
    if (is_initialized != 0) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if (rank != 0) {
        return true;
      }
      std::vector<double> res = input_data_;
      std::ranges::sort(res);
      for (size_t i = 0; i < kCount_; i++) {
        if (res[i] != output_data[i]) {
          return false;
        }
      }
      return true;
    }
    std::vector<double> res = input_data_;
    std::ranges::sort(res);
    for (size_t i = 0; i < kCount_; i++) {
      if (res[i] != output_data[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ZavyalovAQsortPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ZavyalovAQsortMPI, ZavyalovAQsortSEQ>(
    PPC_SETTINGS_zavyalov_a_qsort_simple_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZavyalovAQsortPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ZavyalovAQsortPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace zavyalov_a_qsort_simple_merge
