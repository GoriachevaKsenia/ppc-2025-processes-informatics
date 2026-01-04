#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "../../common/include/common.hpp"
#include "../../mpi/include/ops_mpi.hpp"
#include "../../seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shekhirev_v_custom_reduce {

class ShekhirevVCustomReduceFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    const TestType &params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    if (rank == 0) {
      return output_data == input_data_;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

namespace {

TEST_P(ShekhirevVCustomReduceFuncTests, Test) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestCases = {std::make_tuple(100, "count_100"), std::make_tuple(99, "uneven_count_99"),
                                            std::make_tuple(3, "count_3_less_than_procs"),
                                            std::make_tuple(1, "count_one")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<shekhirev_v_custom_reduce_seq::CustomReduceSequential, InType>(
                       kTestCases, PPC_SETTINGS_shekhirev_v_custom_reduce),
                   ppc::util::AddFuncTask<shekhirev_v_custom_reduce_mpi::CustomReduceMPI, InType>(
                       kTestCases, PPC_SETTINGS_shekhirev_v_custom_reduce));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = ShekhirevVCustomReduceFuncTests::PrintFuncTestName<ShekhirevVCustomReduceFuncTests>;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(CustomReduceTests, ShekhirevVCustomReduceFuncTests, kGtestValues, kTestName);

}  // namespace
}  // namespace shekhirev_v_custom_reduce
