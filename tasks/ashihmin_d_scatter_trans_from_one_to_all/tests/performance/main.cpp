#include <gtest/gtest.h>
#include <mpi.h>

#include <vector>

#include "ashihmin_d_scatter_trans_from_one_to_all/common/include/common.hpp"
#include "ashihmin_d_scatter_trans_from_one_to_all/mpi/include/ops_mpi.hpp"
#include "ashihmin_d_scatter_trans_from_one_to_all/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace ashihmin_d_scatter_trans_from_one_to_all {

class AshihminDRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    ScatterParams params;
    params.elements_per_process = 1000000;
    params.root = 0;
    params.datatype = MPI_INT;

    int world_size = 4;
    int total_elements = params.elements_per_process * world_size;

    params.data.resize(total_elements);
    for (int i = 0; i < total_elements; ++i) {
      params.data[i] = i % 1000;
    }

    input_data_ = params;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

TEST_P(AshihminDRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, AshihminDScatterTransFromOneToAllMPI<int>,
                                                       AshihminDScatterTransFromOneToAllSEQ<int>>(
    PPC_SETTINGS_ashihmin_d_scatter_trans_from_one_to_all);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = AshihminDRunPerfTestsProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, AshihminDRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace ashihmin_d_scatter_trans_from_one_to_all
