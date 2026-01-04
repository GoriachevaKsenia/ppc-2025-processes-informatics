#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "gonozov_l_global_search/common/include/common.hpp"
#include "gonozov_l_global_search/mpi/include/ops_mpi.hpp"
#include "gonozov_l_global_search/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace gonozov_l_global_search {

class GonozovLRunGlobalSearchFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<5>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::make_tuple(std::get<0>(params), std::get<1>(params), std::get<2>(params), std::get<3>(params),
                                  std::get<4>(params));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    auto ordered_data = static_cast<OutType>(std::get<6>(params));
    return std::abs(ordered_data - output_data) < 0.01;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(GonozovLRunGlobalSearchFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {
    std::make_tuple([](double x) { return x * x; }, 2.0, -10.0, 10.0, 0.01, "minimizing_quadratic_function", 0.0),
    std::make_tuple([](double x) { return std::sin(x); }, 2.0, 3.0, 6.0, 0.01, "minimizing_sine", 4.71404),
    std::make_tuple([](double x) { return std::exp(x); }, 2.0, -1.0, 1.0, 0.01, "minimizing_exponent", -0.996669),
    std::make_tuple([](double x) { return ((x * x - 11) * (x * x - 11)) + ((x - 7) * (x - 7)); }, 2.0, -5.0, 5.0, 0.01,
                    "minimizing_himmelblau_function", 3.3965),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<GonozovLGlobalSearchMPI, InType>(kTestParam, PPC_SETTINGS_gonozov_l_global_search),
    ppc::util::AddFuncTask<GonozovLGlobalSearchSEQ, InType>(kTestParam, PPC_SETTINGS_gonozov_l_global_search));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GonozovLRunGlobalSearchFuncTests::PrintFuncTestName<GonozovLRunGlobalSearchFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, GonozovLRunGlobalSearchFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gonozov_l_global_search
