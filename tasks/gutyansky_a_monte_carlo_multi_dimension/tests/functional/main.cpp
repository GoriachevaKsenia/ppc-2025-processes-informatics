#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gutyansky_a_monte_carlo_multi_dimension/common/include/common.hpp"
#include "gutyansky_a_monte_carlo_multi_dimension/mpi/include/ops_mpi.hpp"
#include "gutyansky_a_monte_carlo_multi_dimension/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace gutyansky_a_monte_carlo_multi_dimension {

class GutyanskyAMonteCarloMultiDimensionFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    if (ShouldLoadDataAndTest()) {
      LoadTestDataFromFile();
      return;
    }

    InitializeEmptyData();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double eps = 1e-1;

    if (ShouldLoadDataAndTest()) {
      return std::abs(output_data - output_data_) <= eps;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType output_data_ = {};

  [[nodiscard]] static bool ShouldLoadDataAndTest() {
    const std::string &test_name =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(GetParam());

    if (test_name.find("_mpi") == std::string::npos) {
      return true;
    }

    return !ppc::util::IsUnderMpirun() || ppc::util::GetMPIRank() == 0;
  }

  void InitializeEmptyData() {
    input_data_ = {};
    output_data_ = 0.0;
  }

  void LoadTestDataFromFile() {
    std::string file_name =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam()) + ".txt";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_gutyansky_a_monte_carlo_multi_dimension, file_name);

    std::ifstream ifs(abs_path);

    if (!ifs.is_open()) {
      throw std::runtime_error("Failed to open test file: " + file_name);
    }

    size_t func_id = 0;
    size_t n_dims = 0;
    size_t n_points = 0;
    ifs >> func_id >> n_dims >> n_points;

    std::vector<double> lower_bounds(n_dims);
    std::vector<double> upper_bounds(n_dims);

    for (auto &val : lower_bounds) {
      ifs >> val;
    }

    for (auto &val : upper_bounds) {
      ifs >> val;
    }

    double res = 0.0;
    ifs >> res;

    input_data_ = {.func_id = func_id,
                   .n_dims = n_dims,
                   .lower_bounds = std::move(lower_bounds),
                   .upper_bounds = std::move(upper_bounds),
                   .n_points = n_points};
    output_data_ = res;
  }
};

namespace {

TEST_P(GutyanskyAMonteCarloMultiDimensionFuncTests, MonteCarloMultiDimension) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {"test_1", "test_2", "test_3", "test_4",
                                            "test_5", "test_6", "test_7", "test_8"};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<GutyanskyAMonteCarloMultiDimensionMPI, InType>(
                                               kTestParam, PPC_SETTINGS_gutyansky_a_monte_carlo_multi_dimension),
                                           ppc::util::AddFuncTask<GutyanskyAMonteCarloMultiDimensionSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_gutyansky_a_monte_carlo_multi_dimension));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    GutyanskyAMonteCarloMultiDimensionFuncTests::PrintFuncTestName<GutyanskyAMonteCarloMultiDimensionFuncTests>;

INSTANTIATE_TEST_SUITE_P(MonteCarloTests, GutyanskyAMonteCarloMultiDimensionFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gutyansky_a_monte_carlo_multi_dimension
