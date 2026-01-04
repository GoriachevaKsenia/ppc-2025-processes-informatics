#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kondakov_v_reduce/common/include/common.hpp"
#include "kondakov_v_reduce/mpi/include/ops_mpi.hpp"
#include "kondakov_v_reduce/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kondakov_v_reduce {

class KondakovVReduceRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr std::int64_t kSize = 5'000'000;

 protected:
  void SetUp() override {
    values_.resize(kSize);
    for (std::size_t i = 0; i < values_.size(); ++i) {
      values_[i] = static_cast<int>(i + 1);
    }
    input_data_ = InType{values_, ReduceOp::kSum};
  }

  InType GetTestInputData() override {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output) override {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    if (mpi_initialized != 0) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }

    const std::int64_t expected = kSize * (kSize + 1) / 2;
    return output == expected;
  }

 private:
  std::vector<int> values_;
  InType input_data_;
};

TEST_P(KondakovVReduceRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}
const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KondakovVReduceTaskMPI, KondakovVReduceTaskSEQ>(PPC_SETTINGS_kondakov_v_reduce);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KondakovVReduceRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(KondakovVReduceTests, KondakovVReduceRunPerfTests, kGtestValues, kPerfTestName);
}  // namespace kondakov_v_reduce
