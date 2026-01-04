#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "spichek_d_jacobi/common/include/common.hpp"
#include "spichek_d_jacobi/mpi/include/ops_mpi.hpp"
#include "spichek_d_jacobi/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace spichek_d_jacobi {

class SpichekDJacobiRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) override {
    if (output_data.empty()) {
      return false;
    }

    EXPECT_TRUE(std::ranges::all_of(output_data, [](double v) { return std::isfinite(v); }));

    return true;
  }

  InType GetTestInputData() override {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(SpichekDJacobiRunFuncTestsProcesses, JacobiTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 7> kTestParam = {
    // 1. Базовый тест 2x2
    std::make_tuple(InType{{{10.0, -1.0}, {-1.0, 10.0}}, {9.0, 9.0}, 1e-6, 50}, "simple_2x2"),

    // 2. Базовый тест 3x3
    std::make_tuple(InType{{{4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}}, {6.0, 6.0, 6.0}, 1e-6, 100},
                    "simple_3x3"),

    // 3. Минимальный тест 1x1
    std::make_tuple(InType{{{10.0}}, {20.0}, 1e-6, 10}, "tiny_1x1"),

    // 4. Единичная матрица 3x3
    std::make_tuple(InType{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}, {1.5, -2.5, 3.0}, 1e-6, 100},
                    "identity_3x3"),

    // 5. Тест с нулевым решением
    std::make_tuple(InType{{{5.0, 1.0}, {1.0, 5.0}}, {0.0, 0.0}, 1e-6, 50}, "zero_solution"),

    // 6. Матрица 4x4
    std::make_tuple(InType{{{5.0, 1.0, 0.0, 0.0}, {1.0, 5.0, 1.0, 0.0}, {0.0, 1.0, 5.0, 1.0}, {0.0, 0.0, 1.0, 5.0}},
                           {6.0, 7.0, 7.0, 6.0},
                           1e-6,
                           100},
                    "tridiagonal_4x4"),

    // 7. Матрица 5x5
    std::make_tuple(InType{{{10.0, 1.0, 0.0, 0.0, 0.0},
                            {1.0, 10.0, 1.0, 0.0, 0.0},
                            {0.0, 1.0, 10.0, 1.0, 0.0},
                            {0.0, 0.0, 1.0, 10.0, 1.0},
                            {0.0, 0.0, 0.0, 1.0, 10.0}},
                           {11.0, 12.0, 12.0, 12.0, 11.0},
                           1e-6,
                           100},
                    "band_5x5")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<SpichekDJacobiMPI, InType>(kTestParam, PPC_SETTINGS_spichek_d_jacobi),
                   ppc::util::AddFuncTask<SpichekDJacobiSEQ, InType>(kTestParam, PPC_SETTINGS_spichek_d_jacobi));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = SpichekDJacobiRunFuncTestsProcesses::PrintFuncTestName<SpichekDJacobiRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(JacobiTests, SpichekDJacobiRunFuncTestsProcesses, kGtestValues, kTestName);

}  // namespace

}  // namespace spichek_d_jacobi
