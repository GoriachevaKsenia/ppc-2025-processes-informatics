#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "gonozov_l_simple_iteration_method/common/include/common.hpp"
#include "gonozov_l_simple_iteration_method/mpi/include/ops_mpi.hpp"
#include "gonozov_l_simple_iteration_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace gonozov_l_simple_iteration_method {

class GonozovLRunIterationMethodFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<3>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::tuple<int, std::vector<double>, std::vector<double>> input(static_cast<int>(std::get<0>(params)),
                                                                    std::get<1>(params), std::get<2>(params));
    input_data_ = input;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::vector<double> ordered_data = static_cast<OutType>(std::get<4>(params));
    if (static_cast<int>(ordered_data.size()) != static_cast<int>(output_data.size())) {
      return false;
    }
    for (int i = 0; i < static_cast<int>(std::get<0>(params)); i++) {
      if (std::abs(ordered_data[i] - output_data[i]) > 0.01) {
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
};

namespace {

TEST_P(GonozovLRunIterationMethodFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
    std::make_tuple(3, std::vector<double>{6, 3, 2, 4, 7, 1, 5, 2, 8}, std::vector<double>{10, 5, 3},
                    "system_of_linear_algebraic_equations_containing_three_equations",
                    std::vector<double>{2.1534, -0.3915, -0.8730}),
    std::make_tuple(4, std::vector<double>{7, 1, 3, 2, 5, 13, 4, 2, 5, 6, 21, 4, 7, 2, 3, 15},
                    std::vector<double>{7, 26, 45, 4}, "system_of_linear_algebraic_equations_containing_four_equations",
                    std::vector<double>{0.135702, 1.45937, 1.75892, -0.343029}),
    std::make_tuple(
        5, std::vector<double>{23, 3, 5, 6, 2, 5, 62, 9, 11, 34, 1, 2, 32, 15, 3, 6, 7, 4, 33, 1, 5, 4, 3, 2, 16},
        std::vector<double>{23, 7, 6, 7, 8}, "system_of_linear_algebraic_equations_containing_five_equations",
        std::vector<double>{0.958138, -0.0971273, 0.127852, 0.0370729, 0.196258})};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<GonozovLSimpleIterationMethodMPI, InType>(
                                               kTestParam, PPC_SETTINGS_gonozov_l_simple_iteration_method),
                                           ppc::util::AddFuncTask<GonozovLSimpleIterationMethodSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_gonozov_l_simple_iteration_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GonozovLRunIterationMethodFuncTests::PrintFuncTestName<GonozovLRunIterationMethodFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, GonozovLRunIterationMethodFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gonozov_l_simple_iteration_method
