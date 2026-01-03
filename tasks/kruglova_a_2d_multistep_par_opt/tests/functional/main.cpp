#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "kruglova_a_2d_multistep_par_opt/common/include/common.hpp"
#include "kruglova_a_2d_multistep_par_opt/mpi/include/ops_mpi.hpp"
#include "kruglova_a_2d_multistep_par_opt/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kruglova_a_2d_multistep_par_opt {

class KruglovaA2DMultRunFunkTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  KruglovaA2DMultRunFunkTest() : input_data_(0.0, 0.0, 0.0, 0.0, 0.0, 0) {}

  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::abs(output_data.f_value) < 0.5;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(KruglovaA2DMultRunFunkTest, GlobalOptimizationTask) {
  ExecuteTest(GetParam());
}

const InType kTest1(-5.12, 5.12, -5.12, 5.12, 0.1, 30);
const InType kTest2(-2.0, 2.0, -2.0, 2.0, 0.05, 40);
const InType kTestHighPrecision(-1.0, 1.0, -1.0, 1.0, 0.01, 50);
const InType kTestNarrowRange(0.0, 0.005, 0.0, 0.005, 0.0001, 20);
const InType kTestLargeRange(-10.0, 10.0, -10.0, 10.0, 0.2, 30);
const InType kTestAsymmetric(-5.12, 2.0, -1.0, 5.12, 0.1, 30);

const std::array<TestType, 6> kTestParam = {
    std::make_tuple("Rastrigin_Standard", kTest1),         std::make_tuple("Rastrigin_Narrow", kTest2),
    std::make_tuple("High_Precision", kTestHighPrecision), std::make_tuple("Narrow_Range_Exit", kTestNarrowRange),
    std::make_tuple("Large_Scale", kTestLargeRange),       std::make_tuple("Asymmetric_Range", kTestAsymmetric)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KruglovaA2DMuitMPI, InType>(kTestParam, PPC_SETTINGS_kruglova_a_2d_multistep_par_opt),
    ppc::util::AddFuncTask<KruglovaA2DMuitSEQ, InType>(kTestParam, PPC_SETTINGS_kruglova_a_2d_multistep_par_opt));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KruglovaA2DMultRunFunkTest::PrintFuncTestName<KruglovaA2DMultRunFunkTest>;

INSTANTIATE_TEST_SUITE_P(RastriginTests, KruglovaA2DMultRunFunkTest, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kruglova_a_2d_multistep_par_opt
