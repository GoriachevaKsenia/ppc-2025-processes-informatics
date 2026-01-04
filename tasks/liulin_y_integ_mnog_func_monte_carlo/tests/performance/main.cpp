#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>

#include "liulin_y_integ_mnog_func_monte_carlo/common/include/common.hpp"
#include "liulin_y_integ_mnog_func_monte_carlo/mpi/include/ops_mpi.hpp"
#include "liulin_y_integ_mnog_func_monte_carlo/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace liulin_y_integ_mnog_func_monte_carlo {

class LiulinYIntegMnogFuncMonteCarloPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  struct TestConfig {
    double x_min = 0.0;
    double x_max = 0.0;
    double y_min = 0.0;
    double y_max = 0.0;
    int64_t num_points = 0;
    std::string name;
  };

  static TestConfig GetLiulinSpecificConfig(const std::string &test_name) {
    constexpr int64_t kLargeN = 100000000LL;
    if (test_name.find("liulin_y_integ_mnog_func_monte_carlo_mpi") != std::string::npos) {
      return TestConfig{
          .x_min = 0.0, .x_max = 1.0, .y_min = 0.0, .y_max = 1.0, .num_points = kLargeN, .name = "mpi_large"};
    }
    if (test_name.find("liulin_y_integ_mnog_func_monte_carlo_seq") != std::string::npos) {
      return TestConfig{
          .x_min = 0.0, .x_max = 1.0, .y_min = 0.0, .y_max = 1.0, .num_points = kLargeN, .name = "seq_large"};
    }
    return TestConfig{.x_min = 0.0, .x_max = 1.0, .y_min = 0.0, .y_max = 1.0, .num_points = 100000LL, .name = "small"};
  }

  static bool HasLiulinSpecificConfig(const std::string &test_name) {
    return test_name.find("liulin_y_integ_mnog_func_monte_carlo") != std::string::npos;
  }

  static TestConfig GetTestConfig(const std::string &test_name) {
    if (HasLiulinSpecificConfig(test_name)) {
      return GetLiulinSpecificConfig(test_name);
    }
    if (test_name.find("small") != std::string::npos) {
      return TestConfig{
          .x_min = 0.0, .x_max = 1.0, .y_min = 0.0, .y_max = 1.0, .num_points = 100000LL, .name = "small"};
    }
    if (test_name.find("medium") != std::string::npos) {
      return TestConfig{
          .x_min = 0.0, .x_max = 1.0, .y_min = 0.0, .y_max = 1.0, .num_points = 1000000LL, .name = "medium"};
    }
    if (test_name.find("large") != std::string::npos) {
      return TestConfig{
          .x_min = 0.0, .x_max = 1.0, .y_min = 0.0, .y_max = 1.0, .num_points = 10000000LL, .name = "large"};
    }
    if (test_name.find("xlarge") != std::string::npos) {
      return TestConfig{
          .x_min = 0.0, .x_max = 1.0, .y_min = 0.0, .y_max = 1.0, .num_points = 100000000LL, .name = "xlarge"};
    }
    if (test_name.find("tall") != std::string::npos) {
      return TestConfig{
          .x_min = 0.0, .x_max = 10.0, .y_min = 0.0, .y_max = 0.1, .num_points = 10000000LL, .name = "tall"};
    }
    if (test_name.find("wide") != std::string::npos) {
      return TestConfig{
          .x_min = 0.0, .x_max = 0.1, .y_min = 0.0, .y_max = 10.0, .num_points = 10000000LL, .name = "wide"};
    }
    return TestConfig{.x_min = 0.0, .x_max = 1.0, .y_min = 0.0, .y_max = 1.0, .num_points = 100000LL, .name = "small"};
  }

  void SetUp() override {
    const auto &test_param = GetParam();
    const std::string &test_name = std::get<1>(test_param);

    TestConfig config = GetTestConfig(test_name);
    x_min_ = config.x_min;
    x_max_ = config.x_max;
    y_min_ = config.y_min;
    y_max_ = config.y_max;
    num_points_ = config.num_points;

    GenerateTestData();
  }

  void GenerateTestData() {
    // Тестовая функция f(x,y) = x * y (интеграл = (x_max^2 - x_min^2)/2 * (y_max^2 - y_min^2)/2
    auto f = [](double x, double y) { return x * y; };

    // Используем конструктор TaskInput вместо агрегатной инициализации
    input_data_ = TaskInput(x_min_, x_max_, y_min_, y_max_, f, num_points_);

    // Ожидаемый аналитический интеграл
    expected_output_ = ((x_max_ * x_max_ - x_min_ * x_min_) / 2.0) * ((y_max_ * y_max_ - y_min_ * y_min_) / 2.0);
  }

  bool CheckTestOutputData(OutType &output_data) override {
    constexpr double kRelativeEps = 1e-2;
    if (std::abs(expected_output_) < 1e-10) {
      return std::abs(output_data) < kRelativeEps;
    }
    return std::abs(output_data - expected_output_) / std::abs(expected_output_) < kRelativeEps;
  }

  InType GetTestInputData() override {
    return input_data_;
  }

 private:
  double x_min_ = 0.0;
  double x_max_ = 0.0;
  double y_min_ = 0.0;
  double y_max_ = 0.0;
  int64_t num_points_ = 0;
  InType input_data_;
  OutType expected_output_ = 0.0;
};

TEST_P(LiulinYIntegMnogFuncMonteCarloPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LiulinYIntegMnogFuncMonteCarloSEQ, LiulinYIntegMnogFuncMonteCarloMPI>(
        PPC_SETTINGS_liulin_y_integ_mnog_func_monte_carlo);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LiulinYIntegMnogFuncMonteCarloPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LiulinYIntegMnogFuncMonteCarloPerfTests, kGtestValues, kPerfTestName);

}  // namespace liulin_y_integ_mnog_func_monte_carlo
