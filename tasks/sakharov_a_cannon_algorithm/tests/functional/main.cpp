#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "sakharov_a_cannon_algorithm/common/include/common.hpp"
#include "sakharov_a_cannon_algorithm/mpi/include/ops_mpi.hpp"
#include "sakharov_a_cannon_algorithm/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace ppc::util {
template <typename InType, typename OutType, typename TestType>
static inline void PrintTo(const FuncTestParam<InType, OutType, TestType> &param, ::std::ostream *os) {
  *os << "FuncTestParam{"
      << "name=" << std::get<static_cast<std::size_t>(GTestParamIndex::kNameTest)>(param) << "}";
}
}  // namespace ppc::util

namespace sakharov_a_cannon_algorithm {

class SakharovACannonFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (expected_output_.size() != output_data.size()) {
      return false;
    }
    constexpr double kEps = 1e-9;
    for (std::size_t idx = 0; idx < expected_output_.size(); ++idx) {
      if (std::abs(expected_output_[idx] - output_data[idx]) > kEps) {
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
  OutType expected_output_;
};

namespace {

OutType NaiveMultiply(const InType &input) {
  const int m = input.rows_a;
  const int k = input.cols_a;
  const int n = input.cols_b;
  OutType result(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0);

  for (int ii = 0; ii < m; ++ii) {
    for (int kk = 0; kk < k; ++kk) {
      double a_val = input.a[Idx(k, ii, kk)];
      for (int jj = 0; jj < n; ++jj) {
        result[Idx(n, ii, jj)] += a_val * input.b[Idx(n, kk, jj)];
      }
    }
  }
  return result;
}

TestType MakeCase(const InType &input, const std::string &name) {
  return TestType{input, NaiveMultiply(input), name};
}

TEST_P(SakharovACannonFuncTests, MatrixMultiply) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {
    MakeCase(InType{1, 1, 1, 1, {2.0}, {3.0}}, "single_element"),
    MakeCase(InType{2, 2, 2, 2, {1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}}, "two_by_two"),
    MakeCase(InType{3, 3, 3, 3, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {4, 1, 2, 0, 3, 5, -1, 7, 2}}, "identity_mult"),
    MakeCase(InType{4,
                    4,
                    4,
                    4,
                    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                    {1, 0, 2, -1, 3, 1, 0, 2, 1, 4, -2, 0, 0, 1, 3, 2}},
             "four_by_four"),
    MakeCase(InType{2, 3, 3, 2, {1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}}, "rect_2x3_3x2"),
    MakeCase(InType{3, 2, 2, 4, {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6, 7, 8}}, "rect_3x2_2x4"),
    MakeCase(InType{1, 4, 4, 1, {1, 2, 3, 4}, {5, 6, 7, 8}}, "row_times_col"),
    MakeCase(InType{4, 1, 1, 4, {1, 2, 3, 4}, {5, 6, 7, 8}}, "col_times_row")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SakharovACannonAlgorithmMPI, InType>(kTestParam, PPC_SETTINGS_sakharov_a_cannon_algorithm),
    ppc::util::AddFuncTask<SakharovACannonAlgorithmSEQ, InType>(kTestParam, PPC_SETTINGS_sakharov_a_cannon_algorithm));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SakharovACannonFuncTests::PrintFuncTestName<SakharovACannonFuncTests>;

INSTANTIATE_TEST_SUITE_P(MatrixTests, SakharovACannonFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace sakharov_a_cannon_algorithm
