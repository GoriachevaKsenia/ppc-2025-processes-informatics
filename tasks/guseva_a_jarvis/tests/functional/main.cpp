#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "guseva_a_jarvis/common/include/common.hpp"
#include "guseva_a_jarvis/mpi/include/ops_mpi.hpp"
#include "guseva_a_jarvis/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace guseva_a_jarvis {

class GusevaAJarvisFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string path = ppc::util::GetAbsoluteTaskPath(PPC_ID_guseva_a_jarvis, param + ".txt");
    const auto &[width, height, image, expected] = ReadTestDataFromFile(path);
    input_data_ = std::make_tuple(width, height, image);
    output_data_ = expected;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == output_data_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType output_data_;
};

namespace {

TEST_P(GusevaAJarvisFuncTests, Func) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 11> kTestParam = {"test_0", "test_1", "test_2", "test_3", "test_4", "test_5",
                                             "test_6", "test_7", "test_8", "test_9", "test_10"};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<GusevaAJarvisMPI, InType>(kTestParam, PPC_SETTINGS_guseva_a_jarvis),
                   ppc::util::AddFuncTask<GusevaAJarvisSEQ, InType>(kTestParam, PPC_SETTINGS_guseva_a_jarvis));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GusevaAJarvisFuncTests::PrintFuncTestName<GusevaAJarvisFuncTests>;

INSTANTIATE_TEST_SUITE_P(GusevaAJarvis, GusevaAJarvisFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace guseva_a_jarvis
