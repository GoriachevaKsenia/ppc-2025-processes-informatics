#include <gtest/gtest.h>

#include <array>
#include <cctype>
#include <cstddef>
#include <string>
#include <tuple>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string name = std::get<1>(test_param);
    for (auto &c : name) {
      if (std::isalnum(static_cast<unsigned char>(c)) == 0) {
        c = '_';
      }
    }
    return name;
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const std::string &str_input = std::get<1>(params);
    input_data_.assign(str_input.begin(), str_input.end());
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int expected = 0;
    for (char c : input_data_) {
      if (std::isalpha(static_cast<unsigned char>(c)) != 0) {
        ++expected;
      }
    }
    return expected == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(0, "AbC123"),
    std::make_tuple(1, "onlyLetters"),
    std::make_tuple(2, "1234567890"),
    std::make_tuple(3, "!!!!@@@###"),
    std::make_tuple(4, "a!b@c#1 2 3"),
    std::make_tuple(5, ""),
    std::make_tuple(6, " A \tB \nC "),
    std::make_tuple(7, "aaaaaaaaaabbbbbbbbbbCCCCCCCCCCddddddddddeeeeeeeeeeffffffffff")};

const auto kTasks = std::tuple_cat(
    ppc::util::AddFuncTask<YurkinCountingNumberMPI, InType>(kTestParam, PPC_SETTINGS_yurkin_counting_number),
    ppc::util::AddFuncTask<YurkinCountingNumberSEQ, InType>(kTestParam, PPC_SETTINGS_yurkin_counting_number));

const auto kValues = ppc::util::ExpandToValues(kTasks);
const auto kName = YurkinCountingNumberFuncTests::PrintFuncTestName<YurkinCountingNumberFuncTests>;

TEST_P(YurkinCountingNumberFuncTests, MainTest) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(YurkinTests, YurkinCountingNumberFuncTests, kValues, kName);

}  // namespace
}  // namespace yurkin_counting_number
