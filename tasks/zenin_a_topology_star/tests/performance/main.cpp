#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"
#include "zenin_a_topology_star/common/include/common.hpp"
#include "zenin_a_topology_star/mpi/include/ops_mpi.hpp"
#include "zenin_a_topology_star/seq/include/ops_seq.hpp"

namespace zenin_a_topology_star {

class ZeninATopologyStarPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kMsgSize = 100000000;
  InType input_data_;

  void SetUp() override {
    const auto &full_param = GetParam();
    const std::string &test_name = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kNameTest)>(full_param);
    if (test_name.find("seq_enabled") != std::string::npos) {
      GTEST_SKIP() << "SEQ performance tests are skipped for topology tasks.";
    }
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const int center = 0;
    const int src = center;
    const int dst = (world_size > 1) ? world_size - 1 : 0;
    std::vector<double> data(kMsgSize);
    for (size_t i = 0; i < kMsgSize; ++i) {
      data[i] = static_cast<double>(i);
    }
    input_data_ = std::make_tuple(static_cast<size_t>(src), static_cast<size_t>(dst), std::move(data));
  }
  bool CheckTestOutputData(OutType &output_data) final {
    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    const auto &in = input_data_;
    const int dst = static_cast<int>(std::get<1>(in));
    const auto &data = std::get<2>(in);
    if (world_rank == dst) {
      if (output_data.size() != data.size()) {
        return false;
      }
      for (size_t i = 0; i < data.size(); ++i) {
        if (output_data[i] != data[i]) {
          return false;
        }
      }
    } else {
      if (!output_data.empty()) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ZeninATopologyStarPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ZeninATopologyStarMPI, ZeninATopologyStarSEQ>(
    PPC_SETTINGS_zenin_a_topology_star);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZeninATopologyStarPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(ZeninAPerfTestTopologyStar, ZeninATopologyStarPerfTests, kGtestValues, kPerfTestName);

}  // namespace zenin_a_topology_star
