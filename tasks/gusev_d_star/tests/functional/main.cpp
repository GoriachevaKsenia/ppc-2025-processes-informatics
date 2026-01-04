#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "gusev_d_star/common/include/common.hpp"
#include "gusev_d_star/mpi/include/ops_mpi.hpp"
#include "gusev_d_star/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace gusev_d_star {

class GusevDStarFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    int initialized = 0;
    MPI_Initialized(&initialized);

    if (initialized != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    if (rank == 0) {
      return input_data_ == output_data;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(GusevDStarFuncTests, RunTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 21> kTestParam = {
    // 1. Микро-тесты
    std::make_tuple(1, "Size_1_Min"), std::make_tuple(2, "Size_2_Tiny"),

    // 2. Простые числа
    std::make_tuple(7, "Size_7_Prime"), std::make_tuple(13, "Size_13_Prime"), std::make_tuple(17, "Size_17_Prime"),
    std::make_tuple(29, "Size_29_Prime"), std::make_tuple(41, "Size_41_Prime"), std::make_tuple(53, "Size_53_Prime"),
    std::make_tuple(97, "Size_97_Prime"),

    // 3. Стандартные шаги
    std::make_tuple(10, "Size_10"), std::make_tuple(50, "Size_50"), std::make_tuple(100, "Size_100_Medium"),

    // 4. Степени двойки
    std::make_tuple(32, "Size_32_Pow2"), std::make_tuple(64, "Size_64_Pow2"), std::make_tuple(128, "Size_128_Pow2"),
    std::make_tuple(256, "Size_256_Pow2"),

    // 5. Крупные тесты
    std::make_tuple(150, "Size_150_Large"), std::make_tuple(200, "Size_200_Large"),
    std::make_tuple(250, "Size_250_Large"), std::make_tuple(280, "Size_280_Large"),
    std::make_tuple(300, "Size_300_Max")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<GusevDStarMPI, InType>(kTestParam, PPC_SETTINGS_gusev_d_star),
                   ppc::util::AddFuncTask<GusevDStarSEQ, InType>(kTestParam, PPC_SETTINGS_gusev_d_star));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GusevDStarFuncTests::PrintFuncTestName<GusevDStarFuncTests>;

INSTANTIATE_TEST_SUITE_P(StarTopologyTests, GusevDStarFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gusev_d_star
