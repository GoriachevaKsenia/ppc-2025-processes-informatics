#include <gtest/gtest.h>

#include <array>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "sizov_d_bubble_sort/common/include/common.hpp"
#include "sizov_d_bubble_sort/mpi/include/ops_mpi.hpp"
#include "sizov_d_bubble_sort/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sizov_d_bubble_sort {

class SizovDRunFuncTestsBubbleSort : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return ppc::util::test::SanitizeToken(test_param);
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::string file_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_sizov_d_bubble_sort, param + ".txt");

    std::ifstream file(file_path);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file: " + file_path);
    }

    std::string unsorted_line;
    std::string sorted_line;

    std::getline(file, unsorted_line);
    std::getline(file, sorted_line);
    file.close();

    TrimString(unsorted_line);
    TrimString(sorted_line);

    if (unsorted_line.empty() || sorted_line.empty()) {
      throw std::runtime_error("Input lines cannot be empty in " + file_path);
    }

    std::istringstream unsorted_stream(unsorted_line);
    std::istringstream sorted_stream(sorted_line);

    std::vector<int> unsorted_vec;
    std::vector<int> expected_vec;

    int value = 0;
    while (unsorted_stream >> value) {
      unsorted_vec.push_back(value);
    }
    while (sorted_stream >> value) {
      expected_vec.push_back(value);
    }

    if (unsorted_vec.size() != expected_vec.size()) {
      throw std::runtime_error("Input and expected vectors must have equal size in " + file_path);
    }

    input_data_ = unsorted_vec;
    expected_result_ = expected_vec;
    is_valid_ = true;
  }

  InType GetTestInputData() override {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    if (!is_valid_) {
      return true;
    }
    return output_data == expected_result_;
  }

 private:
  static void TrimString(std::string &s) {
    const auto is_edge_escape = [](unsigned char c) { return c == '\r' || c == '\n' || c == '\t'; };

    std::size_t left = 0;
    while (left < s.size() && is_edge_escape(static_cast<unsigned char>(s[left]))) {
      ++left;
    }

    std::size_t right = s.size();
    while (right > left && is_edge_escape(static_cast<unsigned char>(s[right - 1]))) {
      --right;
    }

    if (left == 0 && right == s.size()) {
      return;
    }

    s.erase(right);
    s.erase(0, left);
  }

  InType input_data_;
  OutType expected_result_;
  bool is_valid_ = true;
};

namespace {

TEST_P(SizovDRunFuncTestsBubbleSort, BubbleSortTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 40> kTestParam = {
    "test1",  "test2",  "test3",  "test4",  "test5",  "test6",  "test7",  "test8",  "test9",  "test10",
    "test11", "test12", "test13", "test14", "test15", "test16", "test17", "test18", "test19", "test20",
    "test21", "test22", "test23", "test24", "test25", "test26", "test27", "test28", "test29", "test30",
    "test31", "test32", "test33", "test34", "test35", "test36", "test37", "test38", "test39", "test40"};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<SizovDBubbleSortMPI, InType>(kTestParam, PPC_SETTINGS_sizov_d_bubble_sort),
                   ppc::util::AddFuncTask<SizovDBubbleSortSEQ, InType>(kTestParam, PPC_SETTINGS_sizov_d_bubble_sort));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = SizovDRunFuncTestsBubbleSort::PrintFuncTestName<SizovDRunFuncTestsBubbleSort>;

INSTANTIATE_TEST_SUITE_P(SizovDBubbleSort, SizovDRunFuncTestsBubbleSort, kGtestValues, kTestName);

}  // namespace

}  // namespace sizov_d_bubble_sort
