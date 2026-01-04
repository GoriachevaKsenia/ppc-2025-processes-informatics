#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "dolov_v_torus_topology/common/include/common.hpp"
#include "dolov_v_torus_topology/mpi/include/ops_mpi.hpp"
#include "dolov_v_torus_topology/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dolov_v_torus_topology {

class DolovVTorusTopologyPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    int world_size = 1;
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init != 0) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    const int k_large_data_size = 10'000'000;
    std::vector<int> big_message(k_large_data_size, 1);

    int sender = 0;
    int receiver = (world_size > 1) ? (world_size - 1) : 0;

    input_data_.sender_rank = sender;
    input_data_.receiver_rank = receiver;
    input_data_.total_procs = world_size;
    input_data_.message = big_message;

    expected_result_.received_message.resize(k_large_data_size, 1);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.received_message.size() != expected_result_.received_message.size()) {
      return false;
    }

    std::size_t size = output_data.received_message.size();
    if (size > 0) {
      if (output_data.received_message[0] != 1 || output_data.received_message[size / 2] != 1 ||
          output_data.received_message[size - 1] != 1) {
        return false;
      }
    }

    return !output_data.route.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_result_;
};

TEST_P(DolovVTorusTopologyPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, DolovVTorusTopologyMPI, DolovVTorusTopologySEQ>(
    PPC_SETTINGS_dolov_v_torus_topology);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

INSTANTIATE_TEST_SUITE_P(TorusPerformance, DolovVTorusTopologyPerfTests, kGtestValues,
                         DolovVTorusTopologyPerfTests::CustomPerfTestName);

}  // namespace dolov_v_torus_topology
