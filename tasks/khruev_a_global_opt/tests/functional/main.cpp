#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "khruev_a_global_opt/common/include/common.hpp"
#include "khruev_a_global_opt/mpi/include/ops_mpi.hpp"
#include "khruev_a_global_opt/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace khruev_a_global_opt {

class KhruevAGlobalOptFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    // auto params = std::get<2>(test_param);
    std::string stroka = std::get<0>(test_param);
    return stroka;
  }

 protected:
  void SetUp() override {
    const auto &param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_.func_id = std::get<1>(param);
    input_data_.ax = std::get<2>(param);
    input_data_.bx = std::get<3>(param);
    input_data_.ay = std::get<4>(param);
    input_data_.by = std::get<5>(param);
    input_data_.epsilon = 0.0001;
    input_data_.max_iter = 100;
    input_data_.r = 4;

    expected_min_ = std::get<6>(param);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double tolerance = 2e-1;
    return std::abs(output_data.value - expected_min_) < tolerance;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  double expected_min_{};
};

namespace {

TEST_P(KhruevAGlobalOptFuncTests, CorrectMinimumFound) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestCases = {
    std::make_tuple("Paraboloid", 1, 0.0, 1.0, 0.0, 1.0, 0.0),
    std::make_tuple("Rastr", 2, 0.0, 1.0, 0.0, 1.0, 0.0),
    std::make_tuple("BoothFunc", 3, -10.0, 10.0, -10.0, 10.0, 0.0),
    std::make_tuple("MatyasFunc", 4, -3.0, 3.0, -3.0, 3.0, 0.0),
    // std::make_tuple("HimmeFunc", 5, 0.0, 1.0, 0.0, 1.0, 0.0)
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<KhruevAGlobalOptMPI, InType>(kTestCases, PPC_SETTINGS_khruev_a_global_opt),
                   ppc::util::AddFuncTask<KhruevAGlobalOptSEQ, InType>(kTestCases, PPC_SETTINGS_khruev_a_global_opt));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = KhruevAGlobalOptFuncTests::PrintFuncTestName<KhruevAGlobalOptFuncTests>;

INSTANTIATE_TEST_SUITE_P(GlobalOptTests, KhruevAGlobalOptFuncTests, kGtestValues, kTestName);

}  // namespace

}  // namespace khruev_a_global_opt
