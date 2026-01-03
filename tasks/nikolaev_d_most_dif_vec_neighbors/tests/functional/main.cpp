#include <gtest/gtest.h>
#include <stb/stb_image.h>

// #include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
// #include <numeric>
// #include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "nikolaev_d_most_dif_vec_neighbors/common/include/common.hpp"
#include "nikolaev_d_most_dif_vec_neighbors/mpi/include/ops_mpi.hpp"
#include "nikolaev_d_most_dif_vec_neighbors/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace nikolaev_d_most_dif_vec_neighbors {

class NikolaevDMostDifVecNeighborsFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    auto expected_ans = std::get<0>(test_param);
    int first = expected_ans.first;
    int second = expected_ans.second;

    std::string str1 = (first >= 0) ? std::to_string(first) : "neg" + std::to_string(-static_cast<int64_t>(first));
    std::string str2 = (second >= 0) ? std::to_string(second) : "neg" + std::to_string(-static_cast<int64_t>(second));
    // return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
    return "expected_ans_" + str1 + "_" + str2;
  }

 protected:
  void SetUp() override {
    // std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_nikolaev_d_most_dif_vec_neighbors, "pic.jpg");
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_ = std::get<1>(params);
    expected_output_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(NikolaevDMostDifVecNeighborsFuncTests, FuncTestsRun) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    std::make_tuple(std::make_pair(0, 0), std::vector<int>{0, 0}),
    std::make_tuple(std::make_pair(1, 3), std::vector<int>{1, 3}),
    std::make_tuple(std::make_pair(3, 15), std::vector<int>{1, 3, 15}),
    std::make_tuple(std::make_pair(1, 32), std::vector<int>{2, 1, 1, 32, 12, 16, 5, -2, -1, -3}),
    std::make_tuple(std::make_pair(-1, 5), std::vector<int>{1, 3, 0, -1, 5, 2}),
    std::make_tuple(std::make_pair(1, 50), std::vector<int>{1, 1, 1, 1, 1, 50, 30, 1, 1}),
    std::make_tuple(std::make_pair(-1, -6), std::vector<int>{1, 2, -1, -6, -2, 2}),
    std::make_tuple(std::make_pair(INT_MIN, INT_MAX),
                    std::vector<int>{-1'000'000'000, 1'000'000'000, INT_MIN, INT_MAX}),
    std::make_tuple(std::make_pair(-200, -50), std::vector<int>{-100, -200, -50, 50, 150, 2}),
    std::make_tuple(std::make_pair(100, 1000), std::vector<int>{350, 1000, 150, 100, 1000})};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<NikolaevDMostDifVecNeighborsMPI, InType>(
                                               kTestParam, PPC_SETTINGS_nikolaev_d_most_dif_vec_neighbors),
                                           ppc::util::AddFuncTask<NikolaevDMostDifVecNeighborsSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_nikolaev_d_most_dif_vec_neighbors));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    NikolaevDMostDifVecNeighborsFuncTests::PrintFuncTestName<NikolaevDMostDifVecNeighborsFuncTests>;

INSTANTIATE_TEST_SUITE_P(MostDifVecNeighbors, NikolaevDMostDifVecNeighborsFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace nikolaev_d_most_dif_vec_neighbors
