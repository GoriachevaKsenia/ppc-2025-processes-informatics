#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "potashnik_m_star_topol/common/include/common.hpp"
#include "potashnik_m_star_topol/mpi/include/ops_mpi.hpp"
#include "potashnik_m_star_topol/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace potashnik_m_star_topol {

class PotashnikMStarTopolFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::vector<int> data;
    data.reserve(params);
    for (int i = 0; i < params; i++) {
      data.push_back(i);
    }
    input_data_ = data;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int type = std::get<1>(output_data);
    if (type == 0) {
      return true;
    }
    int right_res = std::get<0>(output_data);

    int res = 0;
    for (size_t i = 0; i < input_data_.size(); i++) {
      res += input_data_[i];
    }

    return (res == right_res);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(PotashnikMStarTopolFuncTests, StarTopol) {
  ExecuteTest(GetParam());
}

// Test parameters = number of calls
const std::array<TestType, 5> kTestParam = {1, 5, 10, 20, 100};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<PotashnikMStarTopolMPI, InType>(kTestParam, PPC_SETTINGS_potashnik_m_star_topol),
    ppc::util::AddFuncTask<PotashnikMStarTopolSEQ, InType>(kTestParam, PPC_SETTINGS_potashnik_m_star_topol));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = PotashnikMStarTopolFuncTests::PrintFuncTestName<PotashnikMStarTopolFuncTests>;

INSTANTIATE_TEST_SUITE_P(StarTopolTests, PotashnikMStarTopolFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace potashnik_m_star_topol
