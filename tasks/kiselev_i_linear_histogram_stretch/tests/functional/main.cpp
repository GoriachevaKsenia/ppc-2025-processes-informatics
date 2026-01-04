#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "kiselev_i_linear_histogram_stretch/common/include/common.hpp"
#include "kiselev_i_linear_histogram_stretch/mpi/include/ops_mpi.hpp"
#include "kiselev_i_linear_histogram_stretch/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kiselev_i_linear_histogram_stretch {

class KiselevIRunFuncTestsProcesses3 : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType & /*param*/) {
    static int test_counter = 0;
    return "test_" + std::to_string(test_counter++);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    if (rank == 0) {
      return output_data == expected_output_;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(KiselevIRunFuncTestsProcesses3, LinearHistogramStretching) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 13> kTestParam = {
    std::make_tuple(InType{{60, 80, 100, 140, 160}, 5, 1}, OutType{0, 51, 102, 204, 255}, "basic_stretch_alt"),

    std::make_tuple(InType{{0, 40, 120, 200, 255}, 5, 1}, OutType{0, 40, 120, 200, 255}, "full_range_alt"),

    std::make_tuple(InType{{42, 42, 42, 42, 42, 42}, 6, 1}, OutType{42, 42, 42, 42, 42, 42}, "single_color"),

    std::make_tuple(InType{{10, 11}, 2, 1}, OutType{0, 255}, "two_values"),

    std::make_tuple(InType{{50, 75, 100}, 3, 1}, OutType{0, 127, 255}, "non_zero_min"),

    std::make_tuple(InType{{10, 20, 30}, 3, 1}, OutType{0, 128, 255}, "non_255_max"),

    std::make_tuple(InType{{40, 40, 80}, 3, 1}, OutType{0, 0, 255}, "two_same_one_diff"),

    std::make_tuple(InType{{200, 150, 100, 50}, 4, 1}, OutType{255, 170, 85, 0}, "descending_values"),

    std::make_tuple(InType{{254, 255}, 2, 1}, OutType{0, 255}, "min_range"),

    std::make_tuple(InType{{30, 30, 30, 60}, 4, 1}, OutType{0, 0, 0, 255}, "one_outlier"),

    std::make_tuple(InType{{10, 20, 30, 40}, 2, 2}, OutType{0, 85, 170, 255}, "small_2x2"),

    std::make_tuple(InType{{0, 128, 255}, 3, 1}, OutType{0, 128, 255}, "large_step"),

    std::make_tuple(InType{{100, 101, 102, 103}, 4, 1}, OutType{0, 85, 170, 255}, "near_values"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KiselevITestTaskMPI, InType>(kTestParam, PPC_SETTINGS_kiselev_i_linear_histogram_stretch),
    ppc::util::AddFuncTask<KiselevITestTaskSEQ, InType>(kTestParam, PPC_SETTINGS_kiselev_i_linear_histogram_stretch));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KiselevIRunFuncTestsProcesses3::PrintFuncTestName<KiselevIRunFuncTestsProcesses3>;

INSTANTIATE_TEST_SUITE_P(LinearHistogramStretchingTests, KiselevIRunFuncTestsProcesses3, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kiselev_i_linear_histogram_stretch
