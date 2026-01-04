#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "gauss_jordan/common/include/common.hpp"
#include "gauss_jordan/mpi/include/ops_mpi.hpp"
#include "gauss_jordan/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace gauss_jordan {

class RunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<1>(params);
    res_ = std::get<2>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (res_.size() != output_data.size()) {
      return false;
    }
    for (size_t i = 0; i < res_.size(); i++) {
      if (std::abs(res_[i] - output_data[i]) > 1e-6) {
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
  OutType res_;
};

namespace {

TEST_P(RunFuncTestsProcesses, GaussJordan) {
  ExecuteTest(GetParam());
}

// Тестовые системы уравнений
const std::array<TestType, 16> kTestParam = {
    // 1. Единичная матрица 3x3 - самый простой случай
    std::make_tuple("test_identity", std::vector<std::vector<double>>{{1, 0, 0, 1}, {0, 1, 0, 2}, {0, 0, 1, 3}},
                    std::vector<double>{1.0, 2.0, 3.0}),

    // 2. Диагональная матрица 3x3
    std::make_tuple("test_diagonal", std::vector<std::vector<double>>{{2, 0, 0, 4}, {0, 3, 0, 9}, {0, 0, 4, 8}},
                    std::vector<double>{2.0, 3.0, 2.0}),

    // 3. Простая система 2x2 (x + y = 5, 2x - y = 1)
    std::make_tuple("test_simple_2x2", std::vector<std::vector<double>>{{1, 1, 5}, {2, -1, 1}},
                    std::vector<double>{2.0, 3.0}),

    // 4. Система с перестановкой строк
    std::make_tuple("test_row_swap", std::vector<std::vector<double>>{{0, 2, 4}, {1, 1, 3}},
                    std::vector<double>{1.0, 2.0}),

    // 5. Система с одним уравнением
    std::make_tuple("test_single_equation", std::vector<std::vector<double>>{{2, 6}}, std::vector<double>{3.0}),

    // 6. Простая система 4x4 (единичная)
    std::make_tuple(
        "test_4x4_identity",
        std::vector<std::vector<double>>{{1, 0, 0, 0, 1}, {0, 1, 0, 0, 2}, {0, 0, 1, 0, 3}, {0, 0, 0, 1, 4}},
        std::vector<double>{1.0, 2.0, 3.0, 4.0}),

    // 7. Система с отрицательными коэффициентами (-x + y = 1, x + y = 3)
    std::make_tuple("test_negative_coeffs", std::vector<std::vector<double>>{{-1, 1, 1}, {1, 1, 3}},
                    std::vector<double>{1.0, 2.0}),

    // 8. Система с дробными коэффициентами
    std::make_tuple("test_fractional", std::vector<std::vector<double>>{{0.5, 0.25, 1.5}, {0.75, -0.5, 0.5}},
                    std::vector<double>{2.0, 2.0}),

    // 9. Верхнетреугольная система
    std::make_tuple("test_triangular", std::vector<std::vector<double>>{{1, 1, 1, 6}, {0, 1, 1, 5}, {0, 0, 1, 3}},
                    std::vector<double>{1.0, 2.0, 3.0}),

    // 10. Система 5x5 (единичная)
    std::make_tuple(
        "test_5x5_identity",
        std::vector<std::vector<double>>{
            {1, 0, 0, 0, 0, 1}, {0, 1, 0, 0, 0, 2}, {0, 0, 1, 0, 0, 3}, {0, 0, 0, 1, 0, 4}, {0, 0, 0, 0, 1, 5}},
        std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}),

    // 11. Система с большими числами (проверка устойчивости)
    std::make_tuple("test_large_numbers", std::vector<std::vector<double>>{{1000, 2000, 3000}, {2000, 3000, 5000}},
                    std::vector<double>{1.0, 1.0}),

    // 12. Система с очень маленькими числами (близкими к нулю)
    std::make_tuple("test_small_numbers", std::vector<std::vector<double>>{{1e-6, 2e-6, 3e-6}, {2e-6, 3e-6, 5e-6}},
                    std::vector<double>{1.0, 1.0}),

    // 13. Система с почти нулевыми коэффициентами на диагонали (но все же решаемая)
    std::make_tuple("test_nearly_zero_diagonal", std::vector<std::vector<double>>{{1e-8, 1, 1}, {1, 1e-8, 1}},
                    std::vector<double>{1.0, 1.0}),

    // 14. Система 6x6 (единичная для проверки масштабирования)
    std::make_tuple("test_6x6_identity",
                    std::vector<std::vector<double>>{{1, 0, 0, 0, 0, 0, 1},
                                                     {0, 1, 0, 0, 0, 0, 2},
                                                     {0, 0, 1, 0, 0, 0, 3},
                                                     {0, 0, 0, 1, 0, 0, 4},
                                                     {0, 0, 0, 0, 1, 0, 5},
                                                     {0, 0, 0, 0, 0, 1, 6}},
                    std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),

    // 15. Система, требующая перестановки строк (ноль на диагонали)
    std::make_tuple("test_row_permutation_needed",
                    std::vector<std::vector<double>>{{0, 1, 2, 3}, {1, 0, 3, 4}, {2, 3, 0, 5}},
                    std::vector<double>{1.0, 1.0, 1.0}),

    // 16. Система 7x7 (максимальный тест для последовательной версии)
    std::make_tuple("test_7x7_largest",
                    std::vector<std::vector<double>>{{7, 1, 1, 1, 1, 1, 1, 13},
                                                     {1, 6, 1, 1, 1, 1, 1, 12},
                                                     {1, 1, 5, 1, 1, 1, 1, 11},
                                                     {1, 1, 1, 4, 1, 1, 1, 10},
                                                     {1, 1, 1, 1, 3, 1, 1, 9},
                                                     {1, 1, 1, 1, 1, 2, 1, 8},
                                                     {1, 1, 1, 1, 1, 1, 1, 7}},
                    std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0})};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<GaussJordanMPI, InType>(kTestParam, PPC_SETTINGS_gauss_jordan),
                   ppc::util::AddFuncTask<GaussJordanSEQ, InType>(kTestParam, PPC_SETTINGS_gauss_jordan));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = RunFuncTestsProcesses::PrintFuncTestName<RunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(GaussJordanTests, RunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gauss_jordan
