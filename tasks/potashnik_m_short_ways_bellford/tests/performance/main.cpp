#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "potashnik_m_short_ways_bellford/common/include/common.hpp"
#include "potashnik_m_short_ways_bellford/mpi/include/ops_mpi.hpp"
#include "potashnik_m_short_ways_bellford/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace potashnik_m_short_ways_bellford {

class PotashnikMShortWaysBellfordPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 7000;
  InType input_data_;

  void SetUp() override {
    TestType params = kCount_;
    potashnik_m_short_ways_bellford::Graph g = GenerateGraph(params);
    input_data_ = g;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const std::vector<int> &res_dist = output_data;
    potashnik_m_short_ways_bellford::Graph g = input_data_;

    int source = 0;
    std::vector<int> real_dist;
    BellmanFordAlgoSeq(g, source, real_dist);

    if (real_dist.size() != res_dist.size()) {
      return false;
    }

    for (size_t i = 0; i < res_dist.size(); i++) {
      if (res_dist[i] != real_dist[i]) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(PotashnikMShortWaysBellfordPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, PotashnikMShortWaysBellfordMPI, PotashnikMShortWaysBellfordSEQ>(
        PPC_SETTINGS_potashnik_m_short_ways_bellford);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PotashnikMShortWaysBellfordPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PotashnikMShortWaysBellfordPerfTests, kGtestValues, kPerfTestName);

}  // namespace potashnik_m_short_ways_bellford
