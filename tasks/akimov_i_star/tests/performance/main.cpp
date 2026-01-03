#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include "akimov_i_star/common/include/common.hpp"
#include "akimov_i_star/mpi/include/ops_mpi.hpp"
#include "akimov_i_star/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace akimov_i_star {

class AkimovIStarPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const std::size_t messages = 100000;
    std::string s;
    s.reserve(messages * 20);
    for (std::size_t i = 0; i < messages; ++i) {
      s += "send:0:0:hello_from_perf_";
      s += std::to_string(i);
      s += "\n";
    }
    input_data_ = InType(s.begin(), s.end());
    expected_result_ = static_cast<int>(messages);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_result_;
  }

 private:
  InType input_data_;
  OutType expected_result_ = 0;
};

TEST_P(AkimovIStarPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, AkimovIStarMPI, AkimovIStarSEQ>(PPC_SETTINGS_akimov_i_star);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = AkimovIStarPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, AkimovIStarPerfTests, kGtestValues, kPerfTestName);

}  // namespace akimov_i_star
