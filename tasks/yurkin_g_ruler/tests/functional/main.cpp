#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "yurkin_g_ruler/common/include/common.hpp"
#include "yurkin_g_ruler/mpi/include/ops_mpi.hpp"
#include "yurkin_g_ruler/seq/include/ops_seq.hpp"

namespace yurkin_g_ruler {

class YurkinGRulerFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = static_cast<InType>(std::get<0>(params));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (input_data_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(YurkinGRulerFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<YurkinGRulerMPI, InType>(kTestParam, PPC_SETTINGS_yurkin_g_ruler),
                   ppc::util::AddFuncTask<YurkinGRulerSEQ, InType>(kTestParam, PPC_SETTINGS_yurkin_g_ruler));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = YurkinGRulerFuncTests::PrintFuncTestName<YurkinGRulerFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, YurkinGRulerFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace yurkin_g_ruler
