#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "liulin_y_integ_mnog_func_monte_carlo/common/include/common.hpp"
#include "liulin_y_integ_mnog_func_monte_carlo/mpi/include/ops_mpi.hpp"
#include "liulin_y_integ_mnog_func_monte_carlo/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace liulin_y_integ_mnog_func_monte_carlo {

class LiulinYIntegMnogFuncMonteCarloFuncTestsFromFile : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &p) {
    return std::get<1>(p);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::string filename = std::get<1>(params);
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_liulin_y_integ_mnog_func_monte_carlo, filename);

    std::ifstream file(abs_path + ".txt");
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open test file: " + abs_path + ".txt");
    }

    double x_min = 0.0;
    double x_max = 0.0;
    double y_min = 0.0;
    double y_max = 0.0;
    int64_t num_points = 0;
    int func_id = 1;

    file >> x_min >> x_max >> y_min >> y_max >> num_points;

    if (file.good()) {
      file >> func_id;
    }

    std::function<double(double, double)> f;
    double expected = 0.0;

    switch (func_id) {
      case 1:  // f(x,y) = 1
        f = [](double, double) { return 1.0; };
        expected = (x_max - x_min) * (y_max - y_min);
        break;
      case 2:  // f(x,y) = x + y
        f = [](double x, double y) { return x + y; };
        expected = ((((x_max * x_max) - (x_min * x_min)) / 2.0) * (y_max - y_min)) +
                   ((((y_max * y_max) - (y_min * y_min)) / 2.0) * (x_max - x_min));
        break;
      case 3:  // f(x,y) = x * y
        f = [](double x, double y) { return x * y; };
        expected = ((x_max * x_max - x_min * x_min) / 2.0) * ((y_max * y_max - y_min * y_min) / 2.0);
        break;
      case 4:  // f(x,y) = sin(x) * cos(y)
        f = [](double x, double y) { return std::sin(x) * std::cos(y); };
        expected = (std::cos(x_min) - std::cos(x_max)) * (std::sin(y_max) - std::sin(y_min));
        break;
      default:
        throw std::runtime_error("Unknown function id " + std::to_string(func_id) + " in file: " + filename);
    }

    input_data_ = TaskInput(x_min, x_max, y_min, y_max, std::move(f), num_points);
    exp_output_ = expected;

    file.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    constexpr double kAbsEpsSmallN = 0.5;
    constexpr double kAbsEps = 0.05;
    constexpr double kRelEps = 0.05;

    if (input_data_.num_points < 10000LL) {
      return std::abs(output_data - exp_output_) <= kAbsEpsSmallN;
    }
    double area = 0.0;
    area = (input_data_.x_max - input_data_.x_min) * (input_data_.y_max - input_data_.y_min);
    if (std::abs(area) < 1e-10) {
      constexpr double kZeroAreaEps = 0.1;
      return std::abs(output_data) <= kZeroAreaEps;
    }

    if (std::abs(exp_output_) < 1e-4) {
      return std::abs(output_data) <= kAbsEps;
    }
    return std::abs(output_data - exp_output_) / std::abs(exp_output_) <= kRelEps;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType exp_output_ = 0.0;
};

namespace {

TEST_P(LiulinYIntegMnogFuncMonteCarloFuncTestsFromFile, MonteCarloIntegrationFromFile) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    std::make_tuple(0, "const1"),      // f=1
    std::make_tuple(1, "linear"),      // x + y
    std::make_tuple(2, "product"),     // x * y
    std::make_tuple(3, "trig"),        // sin(x)*cos(y)
    std::make_tuple(4, "smallN"),      // мало точек
    std::make_tuple(5, "largeN"),      // много точек
    std::make_tuple(6, "unitSquare"),  // [0,1]x[0,1]
    std::make_tuple(7, "shifted"),     // сдвинутый прямоугольник
    std::make_tuple(8, "negative"),    // отрицательные значения
    std::make_tuple(9, "zeroArea")     // нулевая площадь (ожидаем 0)
};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LiulinYIntegMnogFuncMonteCarloMPI, InType>(
                                               kTestParam, PPC_SETTINGS_liulin_y_integ_mnog_func_monte_carlo),
                                           ppc::util::AddFuncTask<LiulinYIntegMnogFuncMonteCarloSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_liulin_y_integ_mnog_func_monte_carlo));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName =
    LiulinYIntegMnogFuncMonteCarloFuncTestsFromFile::PrintFuncTestName<LiulinYIntegMnogFuncMonteCarloFuncTestsFromFile>;

INSTANTIATE_TEST_SUITE_P(FileTests, LiulinYIntegMnogFuncMonteCarloFuncTestsFromFile, kGtestValues, kFuncTestName);

}  // namespace
}  // namespace liulin_y_integ_mnog_func_monte_carlo
