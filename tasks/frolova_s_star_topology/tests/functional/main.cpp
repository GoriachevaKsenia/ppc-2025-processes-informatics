#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
// #include <utility>

#include "frolova_s_star_topology/common/include/common.hpp"
#include "frolova_s_star_topology/mpi/include/ops_mpi.hpp"
#include "frolova_s_star_topology/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace frolova_s_star_topology {

class FrolovaSRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
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

namespace {

TEST_P(FrolovaSRunFuncTestsProcesses, StarTopologyTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(1, "1"), std::make_tuple(2, "3"), std::make_tuple(3, "4")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<FrolovaSStarTopologyMPI, InType>(kTestParam, PPC_SETTINGS_frolova_s_star_topology),
    ppc::util::AddFuncTask<FrolovaSStarTopologySEQ, InType>(kTestParam, PPC_SETTINGS_frolova_s_star_topology));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = FrolovaSRunFuncTestsProcesses::PrintFuncTestName<FrolovaSRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(StarTopologyTests, FrolovaSRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace frolova_s_star_topology
