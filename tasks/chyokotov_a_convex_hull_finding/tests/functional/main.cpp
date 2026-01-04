#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <climits>
#include <cstddef>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "chyokotov_a_convex_hull_finding/common/include/common.hpp"
#include "chyokotov_a_convex_hull_finding/mpi/include/ops_mpi.hpp"
#include "chyokotov_a_convex_hull_finding/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace chyokotov_a_convex_hull_finding {

class ChyokotovConvexHullFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &matrix = std::get<0>(test_param);
    if (matrix.empty()) {
      return "empty_matrix";
    }
    return "size_of_matrix_" + std::to_string(matrix.size()) + "x" + std::to_string(matrix[0].size());
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    size_t points_actual = 0;
    size_t points_expected = 0;

    for (const auto &i : output_data) {
      points_actual += i.size();
    }

    for (const auto &i : expected_output_) {
      points_expected += i.size();
    }

    if (points_actual != points_expected) {
      return false;
    }

    std::multiset<std::pair<int, int>> set_actual;
    std::multiset<std::pair<int, int>> set_expected;

    for (const auto &i : output_data) {
      set_actual.insert(i.begin(), i.end());
    }

    for (const auto &i : expected_output_) {
      set_expected.insert(i.begin(), i.end());
    }

    return (set_actual == set_expected);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(ChyokotovConvexHullFuncTest, ConvexHullFindingInBinaryImage) {
  ExecuteTest(GetParam());
}

const std::vector<std::vector<int>> kEmptyMatrix = {};
const std::vector<std::vector<std::pair<int, int>>> kExpectedEmpty = {};

const std::vector<std::vector<int>> kOneElement = {{1}};
const std::vector<std::vector<std::pair<int, int>>> kExpectedOneElement = {{{0, 0}}};

const std::vector<std::vector<int>> kWithoutComponents = {{0, 0}, {0, 0}};
const std::vector<std::vector<std::pair<int, int>>> kExpectedWithoutComponents = {};

const std::vector<std::vector<int>> kTwoComponets = {{1, 1, 0, 0}, {1, 1, 0, 0}, {0, 0, 1, 1}, {0, 0, 1, 1}};
const std::vector<std::vector<std::pair<int, int>>> kExpectedTwoComponents = {{{0, 0}, {1, 0}, {0, 1}, {1, 1}},
                                                                              {{2, 2}, {2, 3}, {3, 2}, {3, 3}}};

const std::vector<std::vector<int>> kRectangle = {{0, 0, 0, 0}, {0, 1, 1, 0}, {0, 0, 0, 0}};
const std::vector<std::vector<std::pair<int, int>>> kExpectedRectangle = {{{1, 1}, {2, 1}}};

const std::vector<std::vector<int>> kGrid = {{0, 1, 0}, {1, 0, 1}, {0, 1, 0}, {1, 0, 1}};
const std::vector<std::vector<std::pair<int, int>>> kExpectedGrid = {{{1, 0}}, {{0, 1}}, {{2, 1}},
                                                                     {{1, 2}}, {{2, 3}}, {{0, 3}}};

const std::array<TestType, 6> kTestParam = {
    std::make_tuple(kEmptyMatrix, kExpectedEmpty),
    std::make_tuple(kOneElement, kExpectedOneElement),
    std::make_tuple(kWithoutComponents, kExpectedWithoutComponents),
    std::make_tuple(kTwoComponets, kExpectedTwoComponents),
    std::make_tuple(kRectangle, kExpectedRectangle),
    std::make_tuple(kGrid, kExpectedGrid),
};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ChyokotovConvexHullFindingMPI, InType>(
                                               kTestParam, PPC_SETTINGS_chyokotov_a_convex_hull_finding),
                                           ppc::util::AddFuncTask<ChyokotovConvexHullFindingSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_chyokotov_a_convex_hull_finding));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ChyokotovConvexHullFuncTest::PrintFuncTestName<ChyokotovConvexHullFuncTest>;

INSTANTIATE_TEST_SUITE_P(ConvexHullInBinaryImageTests, ChyokotovConvexHullFuncTest, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace chyokotov_a_convex_hull_finding
