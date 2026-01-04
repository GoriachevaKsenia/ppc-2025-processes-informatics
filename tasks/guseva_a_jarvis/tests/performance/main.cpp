#include <gtest/gtest.h>

#include <string>
#include <tuple>

#include "guseva_a_jarvis/common/include/common.hpp"
#include "guseva_a_jarvis/mpi/include/ops_mpi.hpp"
#include "guseva_a_jarvis/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace guseva_a_jarvis {

class GusevaAJarvisPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType output_data_;

  void SetUp() override {
    std::string path = ppc::util::GetAbsoluteTaskPath(PPC_ID_guseva_a_jarvis, "perf_test.txt");
    const auto &[width, height, image, expected] = ReadTestDataFromFile(path);
    input_data_ = std::make_tuple(width, height, image);
    output_data_ = expected;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(GusevaAJarvisPerfTests, Perf) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GusevaAJarvisMPI, GusevaAJarvisSEQ>(PPC_SETTINGS_guseva_a_jarvis);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GusevaAJarvisPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(GusevaAJarvis, GusevaAJarvisPerfTests, kGtestValues, kPerfTestName);

}  // namespace guseva_a_jarvis
