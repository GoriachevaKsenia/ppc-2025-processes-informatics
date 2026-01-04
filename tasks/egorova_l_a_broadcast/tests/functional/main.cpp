#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "egorova_l_a_broadcast/common/include/common.hpp"
#include "egorova_l_a_broadcast/mpi/include/ops_mpi.hpp"
#include "egorova_l_a_broadcast/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace egorova_l_a_broadcast {

class EgorovaLBroadcastFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto test_params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int mode = std::get<0>(test_params);
    input_data_.root = mode % 2;
    if (mode < 2) {
      input_data_.type_indicator = 0;
      input_data_.data_int = {10, 20, 30};
    } else if (mode < 4) {
      input_data_.type_indicator = 1;
      input_data_.data_float = {1.1F, 2.2F, 3.3F};
    } else {
      input_data_.type_indicator = 2;
      input_data_.data_double = {1.11, 2.22, 3.33};
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (input_data_.type_indicator == 0) {
      return output_data.size() == input_data_.data_int.size() * sizeof(int);
    }
    if (input_data_.type_indicator == 1) {
      return output_data.size() == input_data_.data_float.size() * sizeof(float);
    }
    return output_data.size() == input_data_.data_double.size() * sizeof(double);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(EgorovaLBroadcastFuncTest, RunTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {std::make_tuple(0, "int_r0"),    std::make_tuple(1, "int_r1"),
                                            std::make_tuple(2, "float_r0"),  std::make_tuple(3, "float_r1"),
                                            std::make_tuple(4, "double_r0"), std::make_tuple(5, "double_r1")};

INSTANTIATE_TEST_SUITE_P(
    Broadcast, EgorovaLBroadcastFuncTest,
    ppc::util::ExpandToValues(std::tuple_cat(
        ppc::util::AddFuncTask<EgorovaLBroadcastMPI, InType>(kTestParam, PPC_SETTINGS_egorova_l_a_broadcast),
        ppc::util::AddFuncTask<EgorovaLBroadcastSEQ, InType>(kTestParam, PPC_SETTINGS_egorova_l_a_broadcast))),
    EgorovaLBroadcastFuncTest::PrintFuncTestName<EgorovaLBroadcastFuncTest>);

}  // namespace egorova_l_a_broadcast
