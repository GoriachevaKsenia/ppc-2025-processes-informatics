#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "shvetsova_k_gausse_vert_strip/common/include/common.hpp"
#include "shvetsova_k_gausse_vert_strip/mpi/include/ops_mpi.hpp"
#include "shvetsova_k_gausse_vert_strip/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shvetsova_k_gausse_vert_strip {
static std::pair<std::vector<std::vector<double>>, std::vector<double>> GenerateTestInput(int sz, int size_of_rib);
class ShvetsovaKGaussVertStripRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int sz = 5;           // размер матрицы
    const int size_of_rib = 2;  // полу-ширина + 1 (k = 1)
    input_data_ = GenerateTestInput(sz, size_of_rib);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double eps = 1e-6;

    const auto &matrix = input_data_.first;
    const auto &b = input_data_.second;
    int n = static_cast<int>(matrix.size());

    for (int i = 0; i < n; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        sum += matrix[i][j] * output_data[j];
      }
      if (std::abs(sum - b[i]) > eps) {
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
  OutType expect_res_;
};

static std::pair<std::vector<std::vector<double>>, std::vector<double>> GenerateTestInput(int sz, int size_of_rib) {
  std::vector<std::vector<double>> matrix(sz, std::vector<double>(sz, 0.0));
  std::vector<double> vec(sz, 0.0);

  int k = size_of_rib - 1;

  for (int i = 0; i < sz; ++i) {
    double sum = 0.0;

    for (int j = std::max(0, i - k); j <= std::min(sz - 1, i + k); ++j) {
      if (i == j) {
        continue;
      }
      matrix[i][j] = -1.0;
      sum += std::abs(matrix[i][j]);
    }

    matrix[i][i] = sum + 1.0;  // диагональное преобладание
    vec[i] = 1.0;
  }

  return {matrix, vec};
}

namespace {

TEST_P(ShvetsovaKGaussVertStripRunFuncTestsProcesses, DataFromTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {
    std::make_tuple(InType{std::vector<std::vector<double>>{{2.0, 0.0, 0.0}, {0.0, 4.0, 0.0}, {0.0, 0.0, 5.0}},
                           std::vector<double>{2.0, 8.0, 10.0}},
                    "diag_3_band0"),

    // Тест 2: 4x4 диагональная матрица
    std::make_tuple(InType{std::vector<std::vector<double>>{
                               {3.0, 0.0, 0.0, 0.0}, {0.0, 3.0, 0.0, 0.0}, {0.0, 0.0, 3.0, 0.0}, {0.0, 0.0, 0.0, 3.0}},
                           std::vector<double>{3.0, 6.0, 9.0, 12.0}},
                    "diag_4_band0"),

    // Тест 3: 4x4 ленточная (трёхдиагональная) матрица
    std::make_tuple(
        InType{std::vector<std::vector<double>>{
                   {2.0, -1.0, 0.0, 0.0}, {-1.0, 2.0, -1.0, 0.0}, {0.0, -1.0, 2.0, -1.0}, {0.0, 0.0, -1.0, 2.0}},
               std::vector<double>{1.0, 0.0, 0.0, 1.0}},
        "tri_diag_4_band1"),

    // Тест 4: 5x5 трёхдиагональная
    std::make_tuple(InType{std::vector<std::vector<double>>{{3.0, -1.0, 0.0, 0.0, 0.0},
                                                            {-1.0, 3.0, -1.0, 0.0, 0.0},
                                                            {0.0, -1.0, 3.0, -1.0, 0.0},
                                                            {0.0, 0.0, -1.0, 3.0, -1.0},
                                                            {0.0, 0.0, 0.0, -1.0, 3.0}},
                           std::vector<double>{2.0, 1.0, 0.0, 1.0, 2.0}},
                    "tri_diag_5_band1"),

    // Тест 5: 5x5 диагональная
    std::make_tuple(InType{std::vector<std::vector<double>>{{1.0, 0.0, 0.0, 0.0, 0.0},
                                                            {0.0, 2.0, 0.0, 0.0, 0.0},
                                                            {0.0, 0.0, 3.0, 0.0, 0.0},
                                                            {0.0, 0.0, 0.0, 4.0, 0.0},
                                                            {0.0, 0.0, 0.0, 0.0, 5.0}},
                           std::vector<double>{1.0, 4.0, 9.0, 16.0, 25.0}},
                    "diag_5_band0")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ShvetsovaKGaussVertStripMPI, InType>(kTestParam, PPC_SETTINGS_shvetsova_k_gausse_vert_strip),
    ppc::util::AddFuncTask<ShvetsovaKGaussVertStripSEQ, InType>(kTestParam,
                                                                PPC_SETTINGS_shvetsova_k_gausse_vert_strip));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    ShvetsovaKGaussVertStripRunFuncTestsProcesses::PrintFuncTestName<ShvetsovaKGaussVertStripRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(GaussVertTest, ShvetsovaKGaussVertStripRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace shvetsova_k_gausse_vert_strip
