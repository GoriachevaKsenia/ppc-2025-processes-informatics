#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "kurpiakov_a_shellsort/common/include/common.hpp"
#include "kurpiakov_a_shellsort/mpi/include/ops_mpi.hpp"
#include "kurpiakov_a_shellsort/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kurpiakov_a_shellsort {

class KurpiakovARunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_data_ = std::get<2>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    size_t size_output = output_data.size();
    size_t size_exp = expected_data_.size();

    if (size_output != size_exp) {
      return false;
    }

    for (size_t i = 0; i < size_exp; ++i) {
      if (output_data[i] != expected_data_[i]) {
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
  OutType expected_data_;
};

namespace {

std::vector<int> GenerateRandomVector(int size, int seed = 42) {
  std::vector<int> vec(size);
  std::mt19937 gen(seed);
  std::uniform_int_distribution<> dist(-10000, 10000);
  for (int i = 0; i < size; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}

std::vector<int> GetSortedCopy(const std::vector<int> &vec) {
  std::vector<int> sorted = vec;
  std::ranges::sort(sorted.begin(), sorted.end());
  return sorted;
}

const std::vector<int> kVecEmpty = {};
const std::vector<int> kVecSingle = {42};
const std::vector<int> kVecSorted = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
const std::vector<int> kVecReverse = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
const std::vector<int> kVecDuplicates = {5, 3, 8, 3, 1, 5, 8, 2, 3, 5};
const auto kVec10 = GenerateRandomVector(10);
const auto kVec1000 = GenerateRandomVector(1000);

const std::array<TestType, 7> kTestParam = {
    std::make_tuple(std::make_tuple(0, kVecEmpty), "test1_empty", std::vector<int>{}),
    std::make_tuple(std::make_tuple(1, kVecSingle), "test2_single", std::vector<int>{42}),
    std::make_tuple(std::make_tuple(10, kVecSorted), "test3_sorted", GetSortedCopy(kVecSorted)),
    std::make_tuple(std::make_tuple(10, kVecReverse), "test4_reverse", GetSortedCopy(kVecReverse)),
    std::make_tuple(std::make_tuple(10, kVecDuplicates), "test5_duplicates", GetSortedCopy(kVecDuplicates)),
    std::make_tuple(std::make_tuple(10, kVec10), "test6_random10", GetSortedCopy(kVec10)),
    std::make_tuple(std::make_tuple(1000, kVec1000), "test7_random1000", GetSortedCopy(kVec1000))};

TEST_P(KurpiakovARunFuncTestsProcesses, ShellSortTest) {
  ExecuteTest(GetParam());
}

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KurpiakovAShellsortMPI, InType>(kTestParam, PPC_SETTINGS_kurpiakov_a_shellsort),
    ppc::util::AddFuncTask<KurpiakovAShellsortSEQ, InType>(kTestParam, PPC_SETTINGS_kurpiakov_a_shellsort));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KurpiakovARunFuncTestsProcesses::PrintFuncTestName<KurpiakovARunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(ShellSortTests, KurpiakovARunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kurpiakov_a_shellsort
