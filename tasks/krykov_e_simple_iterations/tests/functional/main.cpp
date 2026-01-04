#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "krykov_e_simple_iterations/common/include/common.hpp"
#include "krykov_e_simple_iterations/mpi/include/ops_mpi.hpp"
#include "krykov_e_simple_iterations/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace krykov_e_simple_iterations {

class KrykovESimpleIterationsFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  KrykovESimpleIterationsFuncTests() = default;

  static std::string PrintTestParam(const TestType &test_param) {
    (void)test_param;
    static int counter = 1;
    return "SLAE_Test_" + std::to_string(counter++);
  }

 protected:
  void SetUp() override {
    const auto &param = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(param);
    expected_output_ = std::get<1>(param);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    constexpr double kEps = 1e-5;
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_output_[i]) > kEps) {
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
  OutType expected_output_;
};

namespace {

// 1) 1x1
const TestType kTest1 = {{1, {4.0}, {8.0}}, {2.0}};

// 2) 2x2
// 4x +  y = 9
//  x + 3y = 5
// решение: (2, 1)
const TestType kTest2 = {{2, {4.0, 1.0, 1.0, 3.0}, {9.0, 5.0}}, {2.0, 1.0}};

// 3) 2x2
// 10x + 2y = 12
//  3x + 8y = 11
// решение: (1, 1)
const TestType kTest3 = {{2, {10.0, 2.0, 3.0, 8.0}, {12.0, 11.0}}, {1.0, 1.0}};

// 4) 3x3
// 10x + y + z = 12
// x + 10y + z = 12
// x + y + 10z = 12
// решение: (1,1,1)
const TestType kTest4 = {{3, {10.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 10.0}, {12.0, 12.0, 12.0}}, {1.0, 1.0, 1.0}};

// 5) 4x4
const TestType kTest5 = {
    {4, {15.0, 2.0, 1.0, 1.0, 1.0, 14.0, 2.0, 1.0, 2.0, 1.0, 13.0, 1.0, 1.0, 1.0, 2.0, 12.0}, {19.0, 18.0, 17.0, 16.0}},
    {1.0, 1.0, 1.0, 1.0}};

TEST_P(KrykovESimpleIterationsFuncTests, SimpleIterationsTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {kTest1, kTest2, kTest3, kTest4, kTest5};
const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KrykovESimpleIterationsMPI, InType>(kTestParam, PPC_SETTINGS_krykov_e_simple_iterations),
    ppc::util::AddFuncTask<KrykovESimpleIterationsSEQ, InType>(kTestParam, PPC_SETTINGS_krykov_e_simple_iterations));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KrykovESimpleIterationsFuncTests::PrintFuncTestName<KrykovESimpleIterationsFuncTests>;

INSTANTIATE_TEST_SUITE_P(SimpleIterationsTests, KrykovESimpleIterationsFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace krykov_e_simple_iterations
