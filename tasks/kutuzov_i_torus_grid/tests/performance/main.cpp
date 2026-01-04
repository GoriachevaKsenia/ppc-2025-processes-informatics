#include <gtest/gtest.h>
#include <mpi.h>

#include <string>
#include <tuple>
#include <vector>

#include "kutuzov_i_torus_grid/common/include/common.hpp"
#include "kutuzov_i_torus_grid/mpi/include/ops_mpi.hpp"
#include "kutuzov_i_torus_grid/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kutuzov_i_torus_grid {
const int kLongStringSize = 10'000'000;

class KutuzovIRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    message_ = std::string(kLongStringSize, 'a');
    int process_count = -1;

    int mpi_used = 0;
    MPI_Initialized(&mpi_used);
    if (mpi_used == 1) {
      MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    } else {
      process_count = 1;
    }

    if (process_count == 1) {
      input_data_ = std::make_tuple(0, 0, message_);
      expected_ = std::make_tuple(std::vector<int>{}, message_);
    } else if (process_count == 2) {
      input_data_ = std::make_tuple(0, 1, message_);
      expected_ = std::make_tuple(std::vector<int>{0, 1}, message_);
    } else if (process_count == 3) {
      input_data_ = std::make_tuple(0, 2, message_);
      expected_ = std::make_tuple(std::vector<int>{0, 2}, message_);
    } else {
      input_data_ = std::make_tuple(0, 3, message_);
      expected_ = std::make_tuple(std::vector<int>{0, 2, 3}, message_);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::vector<int> route = std::get<0>(output_data);
    if (route == seq_route_plug_) {
      return true;
    }

    return expected_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  InType input_data_;
  OutType expected_;

  std::vector<int> seq_route_plug_ = {1, 2, 3, 4, 5};
  std::string message_;
};

TEST_P(KutuzovIRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KutuzovIThorusGridMPI, KutuzovIThorusGridSEQ>(
    PPC_SETTINGS_kutuzov_i_torus_grid);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KutuzovIRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunTests, KutuzovIRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace kutuzov_i_torus_grid
