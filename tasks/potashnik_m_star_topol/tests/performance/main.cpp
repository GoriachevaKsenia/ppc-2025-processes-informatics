#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "potashnik_m_star_topol/common/include/common.hpp"
#include "potashnik_m_star_topol/mpi/include/ops_mpi.hpp"
#include "potashnik_m_star_topol/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace potashnik_m_star_topol {

class PotashnikMStarTopolPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 10000;
  InType input_data_;

  void SetUp() override {
    std::vector<int> data;
    data.reserve(kCount_);
    for (int i = 0; i < kCount_; i++) {
      data.push_back(i);
    }
    input_data_ = data;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int type = std::get<1>(output_data);
    if (type == 0) {
      return true;
    }
    int right_res = std::get<0>(output_data);

    int res = 0;
    for (size_t i = 0; i < input_data_.size(); i++) {
      res += input_data_[i];
    }

    return (res == right_res);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(PotashnikMStarTopolPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, PotashnikMStarTopolMPI, PotashnikMStarTopolSEQ>(
    PPC_SETTINGS_potashnik_m_star_topol);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PotashnikMStarTopolPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PotashnikMStarTopolPerfTests, kGtestValues, kPerfTestName);

}  // namespace potashnik_m_star_topol
