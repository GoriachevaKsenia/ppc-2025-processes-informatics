#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "potashnik_m_short_ways_bellford/common/include/common.hpp"
#include "potashnik_m_short_ways_bellford/mpi/include/ops_mpi.hpp"
#include "potashnik_m_short_ways_bellford/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace potashnik_m_short_ways_bellford {

class PotashnikMShortWaysBellfordFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<2>(GetParam());
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

 private:
  InType input_data_;
};

namespace {

TEST_P(PotashnikMShortWaysBellfordFuncTests, BellFord) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {5, 7, 10, 15, 20};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<PotashnikMShortWaysBellfordMPI, InType>(
                                               kTestParam, PPC_SETTINGS_potashnik_m_short_ways_bellford),
                                           ppc::util::AddFuncTask<PotashnikMShortWaysBellfordSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_potashnik_m_short_ways_bellford));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    PotashnikMShortWaysBellfordFuncTests::PrintFuncTestName<PotashnikMShortWaysBellfordFuncTests>;

INSTANTIATE_TEST_SUITE_P(BellFordTests, PotashnikMShortWaysBellfordFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace potashnik_m_short_ways_bellford
