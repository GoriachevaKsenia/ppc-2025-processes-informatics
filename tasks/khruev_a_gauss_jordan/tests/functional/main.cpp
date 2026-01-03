#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "khruev_a_gauss_jordan/common/include/common.hpp"
#include "khruev_a_gauss_jordan/mpi/include/ops_mpi.hpp"
#include "khruev_a_gauss_jordan/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace khruev_a_gauss_jordan {

class KhruevARunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    return std::get<0>(param);
  }

 protected:
  void SetUp() override {
    TestType test_data = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_matrix_ = std::get<1>(test_data);
    expected_result_ = std::get<2>(test_data);
  }

  bool CheckTestOutputData(OutType &output) final {
    if (expected_result_.size() != output.size()) {
      return false;
    }
    for (size_t i = 0; i < expected_result_.size(); ++i) {
      if (std::abs(expected_result_[i] - output[i]) > 1e-6) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_matrix_;
  }

 private:
  InType input_matrix_;
  OutType expected_result_;
};

namespace {

TEST_P(KhruevARunFuncTestsProcesses, GaussJordan) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 7> kTestParam = {
    // 1) 2x + y = 5
    //    x + y = 4
    //    => x = 1, y = 3
    std::make_tuple("simple_2x2", InType{{2, 1, 5}, {1, 1, 4}}, OutType{1.0, 3.0}),

    // 2) 0x + 3y = 6
    //    2x + y = 5
    //    => y = 2, x = 1.5
    std::make_tuple("row_swap_case", InType{{0, 3, 6}, {2, 1, 5}}, OutType{1.5, 2.0}),

    // 3) x = 4, y = -1, z = 2
    std::make_tuple("identity_3x3", InType{{1, 0, 0, 4}, {0, 1, 0, -1}, {0, 0, 1, 2}}, OutType{4.0, -1.0, 2.0}),

    // 4) 3x = 9
    //    2y = 4
    //    5z = 10
    //    => x = 3, y = 2, z = 2
    std::make_tuple("diagonal_3x3", InType{{3, 0, 0, 9}, {0, 2, 0, 4}, {0, 0, 5, 10}}, OutType{3.0, 2.0, 2.0}),

    // 5) 0.5x + 0.5y = 2
    //    0.5x - 0.5y = 0
    //    => x = 2, y = 2
    std::make_tuple("fractional_coeffs", InType{{0.5, 0.5, 2}, {0.5, -0.5, 0}}, OutType{2.0, 2.0}),

    // 6) 7x = 21
    //    => x = 3
    std::make_tuple("single_equation", InType{{7, 21}}, OutType{3.0}),

    // 7) x + y + z = 6
    //    y + z = 5
    //    z = 3
    //    => x = 1, y = 2, z = 3
    std::make_tuple("upper_triangular", InType{{1, 1, 1, 6}, {0, 1, 1, 5}, {0, 0, 1, 3}}, OutType{1.0, 2.0, 3.0}),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KhruevAGaussJordanMPI, InType>(kTestParam, PPC_SETTINGS_khruev_a_gauss_jordan),
    ppc::util::AddFuncTask<KhruevAGaussJordanSEQ, InType>(kTestParam, PPC_SETTINGS_khruev_a_gauss_jordan));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KhruevARunFuncTestsProcesses::PrintFuncTestName<KhruevARunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(GaussJordanTests, KhruevARunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace
}  // namespace khruev_a_gauss_jordan
