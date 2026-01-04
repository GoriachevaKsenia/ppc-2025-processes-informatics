#include <gtest/gtest.h>

#include <cstddef>
#include <random>
#include <string>

#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"
#include "volkov_a_odd_even_transposition/common/include/common.hpp"
#include "volkov_a_odd_even_transposition/mpi/include/ops_mpi.hpp"
#include "volkov_a_odd_even_transposition/seq/include/ops_seq.hpp"

namespace volkov_a_odd_even_transposition {

class OddEvenSortPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    if (ShouldRun()) {
      const int k_size = 3000;
      input_.resize(k_size);

      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<int> dist(-100000, 100000);

      for (auto &x : input_) {
        x = dist(gen);
      }
    }
  }

  InType GetTestInputData() override {
    return input_;
  }

  bool CheckTestOutputData(OutType &res) override {
    if (ShouldRun()) {
      if (res.size() != input_.size()) {
        return false;
      }
      for (size_t i = 0; i < res.size() - 1; ++i) {
        if (res[i] > res[i + 1]) {
          return false;
        }
      }
    }
    return true;
  }

 private:
  InType input_;

  static bool ShouldRun() {
    const auto &name = std::get<1>(GetParam());
    if (name.find("mpi") != std::string::npos) {
      return !ppc::util::IsUnderMpirun() || ppc::util::GetMPIRank() == 0;
    }
    return true;
  }
};

TEST_P(OddEvenSortPerfTest, RunPipeline) {
  ExecuteTest(GetParam());
}

const auto kPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, OddEvenSortMPI, OddEvenSortSeq>(PPC_SETTINGS_volkov_a_odd_even_transposition);

INSTANTIATE_TEST_SUITE_P(VolkovOddEvenPerf, OddEvenSortPerfTest, ppc::util::TupleToGTestValues(kPerfTasks),
                         OddEvenSortPerfTest::CustomPerfTestName);

}  // namespace volkov_a_odd_even_transposition
