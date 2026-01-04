#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>
#include <tuple>

#include "chaschin_v_broadcast/common/include/common.hpp"
#include "chaschin_v_broadcast/mpi/include/ops_mpi.hpp"
#include "chaschin_v_broadcast/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace chaschin_v_broadcast {

class ChaschinVRunFuncTestsProcessesBR : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);

    input_data_.resize(size);
    for (int i = 0; i < size; ++i) {
      input_data_[i] = (i * 187345543) % 100;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::equal(input_data_.begin(), input_data_.end(), output_data.begin());
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(ChaschinVRunFuncTestsProcessesBR, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    std::make_tuple(1, "1"),   std::make_tuple(3, "3"),   std::make_tuple(5, "5"),   std::make_tuple(7, "7"),
    std::make_tuple(9, "9"),   std::make_tuple(11, "11"), std::make_tuple(13, "13"), std::make_tuple(15, "15"),
    std::make_tuple(17, "17"), std::make_tuple(19, "19")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ChaschinVBroadcastMPI<int>, InType>(kTestParam, PPC_SETTINGS_chaschin_v_broadcast),
    ppc::util::AddFuncTask<ChaschinVBroadcastSEQ<int>, InType>(kTestParam, PPC_SETTINGS_chaschin_v_broadcast));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ChaschinVRunFuncTestsProcessesBR::PrintFuncTestName<ChaschinVRunFuncTestsProcessesBR>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, ChaschinVRunFuncTestsProcessesBR, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace chaschin_v_broadcast
