#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "kutuzov_i_simpson_integration/common/include/common.hpp"
#include "kutuzov_i_simpson_integration/mpi/include/ops_mpi.hpp"
#include "kutuzov_i_simpson_integration/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kutuzov_i_simpson_integration {

class KutuzovIRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int n = get<0>(test_param);
    double x_min = std::get<1>(test_param).first;
    double x_max = std::get<1>(test_param).second;
    double y_min = std::get<2>(test_param).first;
    double y_max = std::get<2>(test_param).second;
    int function_id = std::get<3>(test_param);

    const std::vector<std::string> function_names = {"", "Polynomial", "Trigonometric", "Exponents", "Complex"};
    std::string x_min_str = std::to_string(static_cast<int>(abs(x_min)));
    std::string x_max_str = std::to_string(static_cast<int>(abs(x_max)));
    std::string y_min_str = std::to_string(static_cast<int>(abs(y_min)));
    std::string y_max_str = std::to_string(static_cast<int>(abs(y_max)));

    return std::to_string(n) + "__" + x_min_str + "_" + x_max_str + "__" + y_min_str + "_" + y_max_str + "__" +
           function_names[function_id];
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<int>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = params;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int n = std::get<0>(input_data_);
    double x_min = std::get<1>(input_data_).first;
    double x_max = std::get<1>(input_data_).second;
    double y_min = std::get<2>(input_data_).first;
    double y_max = std::get<2>(input_data_).second;
    int function_id = std::get<3>(input_data_);

    double step_x = (x_max - x_min) / n;
    double step_y = (y_max - y_min) / n;

    double sum = 0.0;
    for (int i = 0; i <= n; i++) {
      double x = x_min + (step_x * i);

      for (int j = 0; j <= n; j++) {
        double y = y_min + (step_y * j);
        double a = GetWeight(i, n) * GetWeight(j, n) * CallFunction(function_id, x, y);
        sum += a;
      }
    }
    sum *= step_x * step_y / 9;
    bool result = (abs(output_data - sum) < 1e-3);
    if (!result) {
      std::cout << '\n' << output_data << ", " << sum << ", " << abs(output_data - sum) << '\n';
    }
    return result;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  static double GetWeight(int i, int n) {
    if (i == 0 || i == n) {
      return 1.0;
    }
    if (i % 2 == 1) {
      return 4.0;
    }
    return 2.0;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(KutuzovIRunFuncTests, SimpsonIntegration) {
  ExecuteTest(GetParam());
}

// n, x_min-x_max, y_min-y_max, function_id
const std::array<TestType, 16> kTestParam = {
    std::make_tuple(2, std::make_pair(0.0, 1.0), std::make_pair(0.0, 1.0), 1),
    std::make_tuple(2, std::make_pair(0.0, 1.0), std::make_pair(0.0, 1.0), 2),
    std::make_tuple(2, std::make_pair(0.0, 1.0), std::make_pair(0.0, 1.0), 3),
    std::make_tuple(2, std::make_pair(0.0, 1.0), std::make_pair(0.0, 1.0), 4),

    std::make_tuple(4, std::make_pair(5.0, 10.0), std::make_pair(5.0, 10.0), 1),
    std::make_tuple(4, std::make_pair(5.0, 10.0), std::make_pair(5.0, 10.0), 2),
    std::make_tuple(4, std::make_pair(5.0, 10.0), std::make_pair(5.0, 10.0), 3),
    std::make_tuple(4, std::make_pair(5.0, 10.0), std::make_pair(5.0, 10.0), 4),

    std::make_tuple(4, std::make_pair(-5.0, 5.0), std::make_pair(-1.0, 1.0), 1),
    std::make_tuple(4, std::make_pair(-5.0, 5.0), std::make_pair(-1.0, 1.0), 2),
    std::make_tuple(4, std::make_pair(-5.0, 5.0), std::make_pair(-1.0, 1.0), 3),
    std::make_tuple(4, std::make_pair(-5.0, 5.0), std::make_pair(-1.0, 1.0), 4),

    std::make_tuple(20, std::make_pair(-1.0, 1.0), std::make_pair(-1.0, 1.0), 1),
    std::make_tuple(20, std::make_pair(-1.0, 1.0), std::make_pair(-1.0, 1.0), 2),
    std::make_tuple(20, std::make_pair(-1.0, 1.0), std::make_pair(-1.0, 1.0), 3),
    std::make_tuple(20, std::make_pair(-1.0, 1.0), std::make_pair(-1.0, 1.0), 4)};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KutuzovISimpsonIntegrationMPI, InType>(
                                               kTestParam, PPC_SETTINGS_kutuzov_i_simpson_integration),
                                           ppc::util::AddFuncTask<KutuzovISimpsonIntegrationSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_kutuzov_i_simpson_integration));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KutuzovIRunFuncTests::PrintFuncTestName<KutuzovIRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, KutuzovIRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kutuzov_i_simpson_integration
