#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "kiselev_i_gauss_method_horizontal_tape_scheme/common/include/common.hpp"
#include "kiselev_i_gauss_method_horizontal_tape_scheme/mpi/include/ops_mpi.hpp"
#include "kiselev_i_gauss_method_horizontal_tape_scheme/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kiselev_i_gauss_method_horizontal_tape_scheme {

class KiselevIRunFuncTestsProcesses2 : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &a_vector = std::get<0>(input_data_);
    const auto &b_vector = std::get<1>(input_data_);

    if (a_vector.empty() || b_vector.empty()) {
      return false;
    }

    const std::size_t num = a_vector.size();
    if (output_data.size() != num) {
      return false;
    }

    double max_abs_residual = 0.0;
    for (std::size_t index = 0; index < num; ++index) {
      double s = 0.0;
      for (std::size_t j_index = 0; j_index < num; ++j_index) {
        s += a_vector[index][j_index] * output_data[j_index];
      }
      const double r_coef = std::abs(s - b_vector[index]);
      max_abs_residual = std::max(max_abs_residual, r_coef);
    }

    return max_abs_residual < 1e-7;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(KiselevIRunFuncTestsProcesses2, GaussHorizontalTapeTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 14> kTestParam = {
    std::make_tuple(InType{std::vector<std::vector<double>>{{-2.0, 0.0, 0.0}, {0.0, -4.0, 0.0}, {0.0, 0.0, -8.0}},
                           std::vector<double>{-2.0, -8.0, -16.0}, std::size_t{0}},
                    "diag_negative_3"),

    std::make_tuple(InType{std::vector<std::vector<double>>{{0.5, 0.0, 0.0}, {0.0, 0.25, 0.0}, {0.0, 0.0, 2.0}},
                           std::vector<double>{1.0, 1.0, 4.0}, std::size_t{0}},
                    "diag_fractional"),

    std::make_tuple(
        InType{std::vector<std::vector<double>>{
                   {4.0, -1.0, 0.0, 0.0}, {-1.0, 4.0, -1.0, 0.0}, {0.0, -1.0, 4.0, -1.0}, {0.0, 0.0, -1.0, 3.0}},
               std::vector<double>{15.0, 10.0, 10.0, 10.0}, std::size_t{1}},
        "tridiag_4_classic"),

    std::make_tuple(InType{std::vector<std::vector<double>>{{5.0, -1.0, -1.0, 0.0, 0.0},
                                                            {-1.0, 6.0, -1.0, -1.0, 0.0},
                                                            {-1.0, -1.0, 7.0, -1.0, -1.0},
                                                            {0.0, -1.0, -1.0, 6.0, -1.0},
                                                            {0.0, 0.0, -1.0, -1.0, 5.0}},
                           std::vector<double>{2.0, 3.0, 4.0, 3.0, 2.0}, std::size_t{2}},
                    "band2_symmetric_5"),

    std::make_tuple(InType{std::vector<std::vector<double>>{
                               {3.0, 1.0, 1.0, 1.0}, {1.0, 3.0, 1.0, 1.0}, {1.0, 1.0, 3.0, 1.0}, {1.0, 1.0, 1.0, 3.0}},
                           std::vector<double>{6.0, 6.0, 6.0, 6.0}, std::size_t{10}},
                    "band_bigger_than_n"),

    std::make_tuple(
        InType{std::vector<std::vector<double>>{{2.0, 1.0}, {1.0, 3.0}}, std::vector<double>{5.0, 5.0}, std::size_t{1}},
        "2x2_simple"),

    std::make_tuple(InType{std::vector<std::vector<double>>{{2.0, 1.0, 1.0}, {0.0, 3.0, 1.0}, {0.0, 0.0, 4.0}},
                           std::vector<double>{8.0, 7.0, 8.0}, std::size_t{2}},
                    "upper_triangular"),

    std::make_tuple(InType{std::vector<std::vector<double>>{{2.0, 0.0, 0.0}, {1.0, 3.0, 0.0}, {1.0, 1.0, 4.0}},
                           std::vector<double>{2.0, 5.0, 7.0}, std::size_t{2}},
                    "lower_triangular"),

    std::make_tuple(InType{std::vector<std::vector<double>>{{5.0, -1.0, 0.0}, {-1.0, 5.0, -1.0}, {0.0, -1.0, 5.0}},
                           std::vector<double>{4.0, 3.0, 4.0}, std::size_t{1}},
                    "alternating_signs"),

    std::make_tuple(InType{std::vector<std::vector<double>>{{1.0}}, std::vector<double>{7.0}, std::size_t{5}},
                    "1x1_large_band"),

    std::make_tuple(InType{std::vector<std::vector<double>>{{10.0, 0.0, 0.0, 0.0, 0.0},
                                                            {0.0, 9.0, 0.0, 0.0, 0.0},
                                                            {0.0, 0.0, 8.0, 0.0, 0.0},
                                                            {0.0, 0.0, 0.0, 7.0, 0.0},
                                                            {0.0, 0.0, 0.0, 0.0, 6.0}},
                           std::vector<double>{10.0, 9.0, 8.0, 7.0, 6.0}, std::size_t{0}},
                    "diag_decreasing"),

    std::make_tuple(InType{std::vector<std::vector<double>>{
                               {6.0, 2.0, 0.0, 0.0}, {1.0, 7.0, 2.0, 0.0}, {0.0, 1.0, 8.0, 2.0}, {0.0, 0.0, 1.0, 9.0}},
                           std::vector<double>{8.0, 10.0, 11.0, 9.0}, std::size_t{1}},
                    "band1_asymmetric"),

    std::make_tuple(InType{std::vector<std::vector<double>>{{3.0, -1.0, 0.0, 0.0, 0.0},
                                                            {-1.0, 3.0, -1.0, 0.0, 0.0},
                                                            {0.0, -1.0, 3.0, -1.0, 0.0},
                                                            {0.0, 0.0, -1.0, 3.0, -1.0},
                                                            {0.0, 0.0, 0.0, -1.0, 3.0}},
                           std::vector<double>{2.0, 1.0, 1.0, 1.0, 2.0}, std::size_t{1}},
                    "laplacian_1d_5"),

    std::make_tuple(InType{std::vector<std::vector<double>>{{4.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                                                            {1.0, 4.0, 1.0, 0.0, 0.0, 0.0},
                                                            {0.0, 1.0, 4.0, 1.0, 0.0, 0.0},
                                                            {0.0, 0.0, 1.0, 4.0, 1.0, 0.0},
                                                            {0.0, 0.0, 0.0, 1.0, 4.0, 1.0},
                                                            {0.0, 0.0, 0.0, 0.0, 1.0, 4.0}},
                           std::vector<double>{5.0, 6.0, 6.0, 6.0, 6.0, 5.0}, std::size_t{1}},
                    "laplacian_1d_6")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KiselevITestTaskMPI, InType>(
                                               kTestParam, PPC_SETTINGS_kiselev_i_gauss_method_horizontal_tape_scheme),
                                           ppc::util::AddFuncTask<KiselevITestTaskSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_kiselev_i_gauss_method_horizontal_tape_scheme));

const auto kGTestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = KiselevIRunFuncTestsProcesses2::PrintFuncTestName<KiselevIRunFuncTestsProcesses2>;

INSTANTIATE_TEST_SUITE_P(GaussTapeTests, KiselevIRunFuncTestsProcesses2, kGTestValues, kPerfTestName);

}  // namespace
}  // namespace kiselev_i_gauss_method_horizontal_tape_scheme
