#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "nikolaev_d_most_dif_vec_neighbors/common/include/common.hpp"
#include "nikolaev_d_most_dif_vec_neighbors/mpi/include/ops_mpi.hpp"
#include "nikolaev_d_most_dif_vec_neighbors/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace nikolaev_d_most_dif_vec_neighbors {

class NikolaevDMostDifVecNeighborsPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_nikolaev_d_most_dif_vec_neighbors, "test_vec.txt");
    std::ifstream file(abs_path);

    if (!file.is_open()) {
      throw std::runtime_error("error");
    }

    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    int number = 0;
    while (iss >> number) {
      input_data_.push_back(number);
    }

    file.close();

    expected_output_ = std::make_pair(-15'000, 15'000);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(NikolaevDMostDifVecNeighborsPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, NikolaevDMostDifVecNeighborsMPI, NikolaevDMostDifVecNeighborsSEQ>(
        PPC_SETTINGS_nikolaev_d_most_dif_vec_neighbors);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NikolaevDMostDifVecNeighborsPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NikolaevDMostDifVecNeighborsPerfTests, kGtestValues, kPerfTestName);

}  // namespace nikolaev_d_most_dif_vec_neighbors
