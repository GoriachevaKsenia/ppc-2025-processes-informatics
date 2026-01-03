#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "sakharov_a_transmission_from_one_to_all/common/include/common.hpp"
#include "sakharov_a_transmission_from_one_to_all/mpi/include/ops_mpi.hpp"
#include "sakharov_a_transmission_from_one_to_all/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sakharov_a_transmission_from_one_to_all {

class SakharovATransmissionFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    InType in = std::get<0>(params);
    input_data_ = in;
    expected_output_ = std::get<1>(in);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(SakharovATransmissionFuncTests, Broadcast) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {
    TestType{InType{0, {1, 2, 3, 4, 5}}, "root_0_simple"},
    TestType{InType{0, {}}, "root_0_empty"},
    TestType{InType{0, {10, 20, 30}}, "root_0_small"},
    TestType{InType{0, std::vector<int>(100, 1)}, "root_0_large"},
};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<SakharovATransmissionFromOneToAllMPI, InType>(
                                               kTestParam, PPC_SETTINGS_sakharov_a_transmission_from_one_to_all),
                                           ppc::util::AddFuncTask<SakharovATransmissionFromOneToAllSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_sakharov_a_transmission_from_one_to_all));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SakharovATransmissionFuncTests::PrintFuncTestName<SakharovATransmissionFuncTests>;

INSTANTIATE_TEST_SUITE_P(BroadcastTests, SakharovATransmissionFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace sakharov_a_transmission_from_one_to_all
