#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "spichek_d_simpson_integral/common/include/common.hpp"
#include "spichek_d_simpson_integral/mpi/include/ops_mpi.hpp"
#include "spichek_d_simpson_integral/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace spichek_d_simpson_integral {

class SpichekDSimpsonIntegralRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    // Количество разбиений (должно быть чётным)
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_ = std::get<0>(params);

    // Аналитическое значение:
    // ∬(x² + y²) dx dy на [0,1]² = 2/3 ≈ 0.666...
    expected_output_ = static_cast<int>(std::round(2.0 / 3.0));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
  OutType expected_output_ = 0;
};

namespace {

TEST_P(SpichekDSimpsonIntegralRunFuncTests, SimpsonIntegral2D) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(10, "10"), std::make_tuple(20, "20"),
                                            std::make_tuple(40, "40")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SpichekDSimpsonIntegralMPI, InType>(kTestParam, PPC_SETTINGS_spichek_d_simpson_integral),
    ppc::util::AddFuncTask<SpichekDSimpsonIntegralSEQ, InType>(kTestParam, PPC_SETTINGS_spichek_d_simpson_integral));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = SpichekDSimpsonIntegralRunFuncTests::PrintFuncTestName<SpichekDSimpsonIntegralRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(SimpsonIntegralFuncTests, SpichekDSimpsonIntegralRunFuncTests, kGtestValues, kTestName);

}  // namespace

}  // namespace spichek_d_simpson_integral
