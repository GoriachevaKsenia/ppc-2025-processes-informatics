#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "fatehov_k_reshetka_tor/common/include/common.hpp"
#include "fatehov_k_reshetka_tor/mpi/include/ops_mpi.hpp"
#include "fatehov_k_reshetka_tor/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace fatehov_k_reshetka_tor {

class FatehovKRunFuncTestsReshetkaTor : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string out = std::to_string(std::get<0>(test_param)) + "_matrix_" + std::to_string(std::get<1>(test_param)) +
                      "x" + std::to_string(std::get<2>(test_param));
    std::ranges::replace(out, '-', 'm');
    std::ranges::replace(out, '.', '_');
    return out;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    size_t rows = std::get<1>(params);
    size_t columns = std::get<2>(params);
    std::vector<double> matrix = std::get<3>(params);

    double expected = ComputeExpectedValue(matrix);

    input_data_ = std::make_tuple(rows, columns, matrix);
    expected_result_ = expected;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::fabs(expected_result_ - output_data) < 1e-10;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  static double ComputeExpectedValue(const std::vector<double> &matrix) {
    double global_max = -1e18;
    for (double val : matrix) {
      double heavy_val = val;
      for (int k = 0; k < 100; ++k) {
        heavy_val = (std::sin(heavy_val) * std::cos(heavy_val)) + std::exp(std::complex<double>(0, heavy_val).real()) +
                    std::sqrt(std::abs(heavy_val) + 1.0);
        if (std::isinf(heavy_val)) {
          heavy_val = val;
        }
      }
      global_max = std::max(heavy_val, global_max);
    }
    return global_max;
  }

  InType input_data_ = std::make_tuple(0, 0, std::vector<double>{});
  OutType expected_result_ = 0;
};

namespace {

TEST_P(FatehovKRunFuncTestsReshetkaTor, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
    std::make_tuple(1, 3, 4, std::vector<double>{1.3, 2.4, 3.1, 4, 5.0, 6.2, 7, 8, 9, 10, 11, 12}, 0),
    std::make_tuple(2, 3, 3, std::vector<double>{-10.3, -9.1, -8.5, -7.1, -6, -5, -4.0, -3, -2}, 0),
    std::make_tuple(3, 5, 5, std::vector<double>{10000,      124124, 65789, 75445, 4123412, 1412412, 56,  65,  -2,
                                                 -151234124, 63124,  454,   223,   454,     232,     565, 232, 7878,
                                                 2324,       57546,  12412, 454,   434,     466,     444},
                    0)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<FatehovKReshetkaTorMPI, InType>(kTestParam, PPC_SETTINGS_fatehov_k_reshetka_tor),
    ppc::util::AddFuncTask<FatehovKReshetkaTorSEQ, InType>(kTestParam, PPC_SETTINGS_fatehov_k_reshetka_tor));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = FatehovKRunFuncTestsReshetkaTor::PrintFuncTestName<FatehovKRunFuncTestsReshetkaTor>;

INSTANTIATE_TEST_SUITE_P(TestReshetkaTor, FatehovKRunFuncTestsReshetkaTor, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace fatehov_k_reshetka_tor
