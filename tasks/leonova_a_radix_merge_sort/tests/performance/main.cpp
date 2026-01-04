#include <gtest/gtest.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include "leonova_a_radix_merge_sort/common/include/common.hpp"
#include "leonova_a_radix_merge_sort/mpi/include/ops_mpi.hpp"
#include "leonova_a_radix_merge_sort/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace leonova_a_radix_merge_sort {

class LeonovaARunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  std::vector<TestType> test_data_;
  size_t current_test_index_ = 0;

  void SetUp() override {
    test_data_ = ReadTestDataFromFile("tasks/leonova_a_radix_merge_sort/data/test_data.txt");
    current_test_index_ = 0;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    OutType expected_output = std::get<1>(test_data_[current_test_index_ - 1]);
    return output_data == expected_output;
  }

  InType GetTestInputData() final {
    if (current_test_index_ < test_data_.size()) {
      return std::get<0>(test_data_[current_test_index_++]);
    }
    return std::get<0>(test_data_.back());
  }

 private:
  static std::vector<TestType> ReadTestDataFromFile(const std::string &filepath) {
    std::vector<TestType> result;

    std::ifstream file(filepath);
    if (!file.is_open()) {
      return result;
    }

    int num_tests = 0;
    file >> num_tests;

    for (int test_index = 0; test_index < num_tests; test_index++) {
      int input_size = 0;
      file >> input_size;

      InType input_array(input_size);
      for (int i = 0; i < input_size; i++) {
        file >> input_array[i];
      }

      int output_size = 0;
      file >> output_size;

      OutType expected_array(output_size);
      for (int i = 0; i < output_size; i++) {
        file >> expected_array[i];
      }

      result.emplace_back(input_array, expected_array);
    }

    file.close();
    return result;
  }
};

TEST_P(LeonovaARunPerfTests, RunPerfRadix) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, LeonovaARadixMergeSortMPI, LeonovaARadixMergeSortSEQ>(
    PPC_SETTINGS_leonova_a_radix_merge_sort);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LeonovaARunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunPerfRadixTests, LeonovaARunPerfTests, kGtestValues, kPerfTestName);

}  // namespace leonova_a_radix_merge_sort
