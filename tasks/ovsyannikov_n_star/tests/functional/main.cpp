#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "ovsyannikov_n_star/common/include/common.hpp"
#include "ovsyannikov_n_star/mpi/include/ops_mpi.hpp"
#include "ovsyannikov_n_star/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace ovsyannikov_n_star {

class OvsyannikovNStarFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data >= 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 1;
};

TEST_P(OvsyannikovNStarFuncTests, StarTopologyTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(1, "1"), std::make_tuple(2, "3"), std::make_tuple(3, "4")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<OvsyannikovNStarMPI, InType>(kTestParam, PPC_SETTINGS_ovsyannikov_n_star),
                   ppc::util::AddFuncTask<OvsyannikovNStarSEQ, InType>(kTestParam, PPC_SETTINGS_ovsyannikov_n_star));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = OvsyannikovNStarFuncTests::PrintFuncTestName<OvsyannikovNStarFuncTests>;

INSTANTIATE_TEST_SUITE_P(StarTopologyTests, OvsyannikovNStarFuncTests, kGtestValues, kPerfTestName);

}  // namespace ovsyannikov_n_star
